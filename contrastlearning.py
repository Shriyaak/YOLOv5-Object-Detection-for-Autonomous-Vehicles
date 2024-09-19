import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from datafetch import KittiDataset, collate_fn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.amp import autocast
import os


# Defining the  CIoU loss function
def ciou_loss(pred_boxes, target_boxes):
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return torch.tensor(0.0).to(pred_boxes.device if pred_boxes else 'cpu')

    def bbox_iou(box1, box2):
        inter_x1 = torch.max(box1[0], box2[0])
        inter_y1 = torch.max(box1[1], box2[1])
        inter_x2 = torch.min(box1[2], box2[2])
        inter_y2 = torch.min(box1[3], box2[3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        return iou

    def ciou(box1, box2):
        iou = bbox_iou(box1, box2)
        center_dist = torch.sqrt(
            (box1[0] + box1[2] - box2[0] - box2[2]) ** 2 + (box1[1] + box1[3] - box2[1] - box2[3]) ** 2)
        ciou = iou - (center_dist / (box1[2] - box1[0] + box2[2] - box2[0] + 1e-6))
        return 1 - ciou

    loss = 0
    for pred, target in zip(pred_boxes, target_boxes):
        loss += ciou(pred, target)
    return loss / len(pred_boxes) if len(pred_boxes) > 0 else torch.tensor(0.0)


def preprocess_student_outputs(outputs):
    processed_outputs = []
    for output in outputs:
        if isinstance(output, torch.Tensor):
            if output.dim() == 5:
                output = output.mean(dim=-1)
            elif output.dim() == 4:
                if output.shape[1] > 0:
                    processed_outputs.append(output)
            elif output.dim() == 2:
                if output.shape[1] > 0:
                    processed_outputs.append(output.unsqueeze(0))
            else:
                raise ValueError(f"Unexpected output dimension: {output.dim()}")
        else:
            raise TypeError("Output is not a tensor")

    return processed_outputs


def find_min_common_size(feature_maps):
    sizes = [o.shape[-2:] for o in feature_maps]
    min_size = [min(s) for s in zip(*sizes)]
    return tuple(int(dim) for dim in min_size)


def resize_features(features, target_size):
    resized_features = []
    for feature in features:
        if feature.dim() == 4:
            if feature.shape[2] == 0 or feature.shape[3] == 0:
                continue
            resized_feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(resized_feature)
        elif feature.dim() == 3:
            feature = feature.unsqueeze(0)
            if feature.shape[2] == 0 or feature.shape[3] == 0:
                continue
            resized_feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(resized_feature.squeeze(0))
        else:
            raise ValueError(f"Unexpected feature dimension for resizing: {feature.dim()}")
    return resized_features


class FeatureAggregator(nn.Module):
    def __init__(self, aggregation_type='concat'):
        super(FeatureAggregator, self).__init__()
        self.aggregation_type = aggregation_type

    def forward(self, x):
        x = [layer(x) for layer in self.model_layers]
        sizes = [xi.shape for xi in x]
        assert all(size == sizes[0] for size in sizes[1:]), "Mismatch in tensor sizes before concatenation"

        if self.aggregation_type == 'concat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggregation_type}")


class MeanTeacher:
    def __init__(self, student_model, teacher_model, alpha=0.99):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.alpha = alpha

    def update_teacher(self):
        student_dict = self.student_model.state_dict()
        teacher_dict = self.teacher_model.state_dict()
        for k in teacher_dict.keys():
            teacher_dict[k] = self.alpha * teacher_dict[k] + (1 - self.alpha) * student_dict[k]
        self.teacher_model.load_state_dict(teacher_dict)


def inspect_yolov5_output(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")


def load_model(model_path):
    student_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    teacher_model = copy.deepcopy(student_model)
    return student_model, teacher_model


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=-1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.to(similarity_matrix.device)
        labels = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = labels.float()

        exp_sim = torch.exp(similarity_matrix)
        loss = -torch.log(exp_sim / torch.sum(exp_sim, dim=-1, keepdim=True))
        loss = (mask * loss).sum() / mask.sum()
        return loss


class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, pseudo_labels, transform=None):
        self.image_paths = image_paths
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (416, 416))  # Providing a default image if loading fails

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = torch.tensor(self.pseudo_labels[idx], dtype=torch.float32)
        return {'image': image, 'label': label}


def generate_pseudo_labels(teacher_model, train_dataset, threshold=0.3):
    image_paths = []
    pseudo_labels = []

    for img_filename in train_dataset.img_filenames:
        image_path = os.path.join(train_dataset.img_dir, img_filename + '.png')

        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(next(teacher_model.parameters()).device)
        with torch.no_grad():
            outputs = teacher_model(image_tensor)
            predictions = outputs[0] if isinstance(outputs, list) else outputs
            boxes = extract_boxes(predictions, conf_threshold=threshold)

        if boxes:
            image_paths.append(image_path)
            pseudo_labels.append(boxes)

    return image_paths, pseudo_labels


def extract_boxes(predictions, conf_threshold=0.4):
    pred = predictions[predictions[..., 4] > conf_threshold]

    boxes = []
    for det in pred:
        x_center, y_center, width, height = det[:4]
        conf = det[4]
        boxes.append([x_center.item(), y_center.item(), width.item(), height.item(), conf.item()])

    return boxes

def main():
    alpha = 0.99
    learning_rate = 0.001
    num_epochs = 50

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])

    train_dataset = KittiDataset(
        img_dir='/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/images1/train',
        label_dir='/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/images1/train_labels',
        transform=transform
    )
    pseudo_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    student_model, teacher_model = load_model(
        '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/runs/train/kiti_yolov5/weights/best.pt'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()
    mean_teacher = MeanTeacher(student_model, teacher_model, alpha=alpha)

    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    contrastive_loss_fn = ContrastiveLoss()

    total_pseudo_labels_count = 0

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        for batch in pseudo_loader:
            images = batch['image'].to(device)
            targets = batch['label'].to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                student_outputs = student_model(images)
                student_outputs = preprocess_student_outputs(student_outputs)
                target_features = preprocess_student_outputs(targets)

                if len(student_outputs) == 0 or len(target_features) == 0:
                    continue

                min_size = find_min_common_size(student_outputs)
                student_outputs = resize_features(student_outputs, min_size)
                target_features = resize_features(target_features, min_size)

                feature_aggregator = FeatureAggregator('concat')
                aggregated_features = feature_aggregator(student_outputs)

                loss = contrastive_loss_fn(aggregated_features, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        mean_teacher.update_teacher()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(pseudo_loader)}")

        image_paths, pseudo_labels = generate_pseudo_labels(teacher_model, train_dataset, threshold=0.3)
        total_pseudo_labels_count += len(pseudo_labels)

        pseudo_dataset = PseudoLabelDataset(image_paths, pseudo_labels, transform=transform)
        pseudo_loader = DataLoader(pseudo_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    print(f"Total number of pseudo labels created: {total_pseudo_labels_count}")


if __name__ == "__main__":
    main()






