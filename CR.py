import torch
import torch.nn as nn
import torch.optim as optim
import copy
from datafetch import KittiDataset, collate_fn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Defining loss function CIoU:
def ciou_loss(pred_boxes, target_boxes):
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return torch.tensor(0.0).to(pred_boxes.device if pred_boxes else 'cpu')

    def bbox_iou(pred_boxes, target_boxes):
        """Compute Intersection over Union (IoU) between two sets of boxes."""
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.split(1, dim=-1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.split(1, dim=-1)

        # Intersection area
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Union area
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        return iou

    # Computing IoU
    iou = bbox_iou(pred_boxes, target_boxes)

    # CIoU components
    pred_cx = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
    pred_cy = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
    pred_w = pred_boxes[..., 2] - pred_boxes[..., 0]
    pred_h = pred_boxes[..., 3] - pred_boxes[..., 1]

    target_cx = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
    target_cy = (target_boxes[..., 1] + target_boxes[..., 3]) / 2
    target_w = target_boxes[..., 2] - target_boxes[..., 0]
    target_h = target_boxes[..., 3] - target_boxes[..., 1]

    # Diagonal of the smallest enclosing box
    enclose_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
    enclose_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
    enclose_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
    enclose_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    enclose_diagonal = torch.sqrt(enclose_w ** 2 + enclose_h ** 2)

    # Center distance
    center_distance = torch.sqrt((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2)

    # CIoU Loss
    ciou = iou - (center_distance / torch.clamp(enclose_diagonal, min=1e-6)) + (1 - iou)

    return 1 - ciou.mean()


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.conv1(x)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map


# Defining the Mean Teacher class
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


class YOLOv5WithAttention(nn.Module):
    def __init__(self, base_model, in_channels):
        super(YOLOv5WithAttention, self).__init__()
        self.base_model = base_model
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        features = self.base_model(x)
        features_with_attention = self.spatial_attention(features)
        return self.base_model.predict(features_with_attention)


# Loading the YOLOv5 model
def load_model(model_path):
    student_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    teacher_model = copy.deepcopy(student_model)
    return student_model, teacher_model


def extract_boxes(predictions):
    """
    Extract bounding boxes from model predictions.

    Args:
        predictions (list of torch.Tensor): Model outputs with shapes corresponding to different scales.

    Returns:
        torch.Tensor: Extracted bounding boxes.
    """
    all_boxes = []
    batch_size = predictions[0].size(0)

    for pred in predictions:
        # Each tensor is of shape [batch_size, num_anchors, grid_size, grid_size, num_attributes]
        pred_boxes = pred.view(batch_size, -1, 9)
        all_boxes.append(pred_boxes)

    # Concatenating all scale predictions
    boxes = torch.cat(all_boxes, dim=1)
    return boxes


def convert_student_outputs(student_outputs, num_boxes, num_classes):
    batch_size = student_outputs.size(0)

    # Reshaping student outputs to have dimensions [batch_size, num_boxes, num_elements]
    student_outputs = student_outputs.view(batch_size, num_boxes, -1)

    # Checking if the reshaped student outputs have the correct number of elements
    num_elements = 4 + 1 + num_classes
    if student_outputs.size(2) != num_elements:
        raise ValueError(f"Expected {num_elements} elements per box but found {student_outputs.size(2)}")

    return student_outputs


def convert_teacher_outputs_to_yolo_format(teacher_outputs, num_classes):
    batch_size = len(teacher_outputs)
    num_boxes = teacher_outputs[0].size(0)

    # Initializing  tensor to hold YOLO format outputs
    yolo_teacher_outputs = torch.zeros(batch_size, num_boxes, 4 + 1 + num_classes)

    for i, output in enumerate(teacher_outputs):
        # Assuming each output tensor has the shape [num_boxes, 9]
        assert output.size(1) == 9, f"Unexpected size {output.size(1)} in output. Expected 9."
        yolo_output = torch.zeros(num_boxes, 4 + 1 + num_classes)
        yolo_output[:, :4] = output[:, :4]
        yolo_output[:, 4] = output[:, 4]
        class_scores = torch.zeros(num_boxes, num_classes)
        class_scores[:, :output.size(1) - 5] = output[:, 5:]  # Adjusting class scores

        yolo_output[:, 5:] = class_scores

        yolo_teacher_outputs[i] = yolo_output

    return yolo_teacher_outputs


def align_outputs(student_outputs, teacher_outputs, num_classes):

    # Processing student outputs to YOLO format
    student_boxes = torch.cat([output.view(output.size(0), -1) for output in student_outputs], dim=1)

    # Converting teacher outputs to YOLO format
    teacher_boxes = convert_teacher_outputs_to_yolo_format(teacher_outputs, num_classes)
    teacher_boxes = torch.cat([box.view(box.size(0), -1) for box in teacher_boxes], dim=1)

    return student_boxes, teacher_boxes

def flatten_yolo_outputs(yolo_output):

    batch_size = yolo_output.size(0)
    num_boxes = yolo_output.size(1)
    num_elements = yolo_output.size(2)
    return yolo_output.view(batch_size, num_boxes, num_elements)

def compute_loss(pred_boxes, targets):

    # Ensure shapes match
    assert pred_boxes.shape[1:] == targets.shape[1:], "Mismatch in the shape of predictions and targets."
    loss = torch.mean((pred_boxes - targets) ** 2)

    return loss

def pad_or_truncate(tensor, num_boxes):
    batch_size, current_num_boxes, num_elements = tensor.size()
    if current_num_boxes < num_boxes:
        padding = (0, 0, 0, num_boxes - current_num_boxes)
        tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
    elif current_num_boxes > num_boxes:
        tensor = tensor[:, :num_boxes, :]
    return tensor


def main():
    # Paths to data
    train_img_dir_labeled = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/images1/train'
    train_lbl_dir_labeled = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/Dataset_original/train/labels'
    train_img_dir_unlabeled = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/images1/unlabeled'
    val_img_dir = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/images1/val'
    val_lbl_dir = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/Dataset_original/val/labels'

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    default_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initializing datasets
    labeled_dataset = KittiDataset(train_img_dir_labeled, train_lbl_dir_labeled, transform=train_transform)
    unlabeled_dataset = KittiDataset(train_img_dir_unlabeled, transform=default_transform)
    val_dataset = KittiDataset(val_img_dir, val_lbl_dir, transform=default_transform)

    # Data loaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Loading YOLOv5 model
    model_path = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/runs/train/kiti_yolov5/weights/best.pt'
    student_model, teacher_model = load_model(model_path)

    # Transfering models to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_model.to(device)
    teacher_model.to(device)

    # Initializing MeanTeacher and optimizer
    mean_teacher = MeanTeacher(student_model, teacher_model)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    # Training settings
    epochs = 10

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0

        student_model.train()
        teacher_model.eval()

        for (batch_labeled, batch_unlabeled) in zip(labeled_loader, unlabeled_loader):
            # Loading labeled and unlabeled data
            labeled_images, labeled_targets = batch_labeled['image'], batch_labeled['label']
            unlabeled_images = batch_unlabeled['image']

            # Moving data to the appropriate device
            labeled_images, labeled_targets = labeled_images.to(device), labeled_targets.to(device)
            unlabeled_images = unlabeled_images.to(device)

            # Forward pass for the student model
            student_outputs = student_model(labeled_images)
            print(f"Student outputs shape: {[out.shape for out in student_outputs]}")

            # Forward pass for the teacher model with no gradient calculation
            with torch.no_grad():
                teacher_outputs = teacher_model(unlabeled_images)
            print(f"Teacher outputs shape: {[out.shape for out in teacher_outputs]}")

            # Converting and aligning outputs
            student_boxes = extract_boxes(student_outputs)
            teacher_boxes = convert_teacher_outputs_to_yolo_format(teacher_outputs, num_classes=80)

            print(f"Student boxes shape: {student_boxes.shape}")
            print(f"Labeled targets shape: {labeled_targets.shape}")

            # Compute losses
            labeled_loss = compute_loss(student_boxes, labeled_targets)
            pseudo_labels = teacher_boxes

            print(f"Unlabeled student outputs shape: {student_boxes.shape}")
            print(f"Pseudo labels shape: {pseudo_labels.shape}")

            unlabeled_loss = compute_loss(student_boxes, pseudo_labels)

            # Total loss combining labeled and unlabeled data
            loss = labeled_loss + 0.5 * unlabeled_loss
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Updating teacher model using Mean Teacher or similar approach
            mean_teacher.update_teacher()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(labeled_loader)}")


if __name__ == "__main__":
    main()



