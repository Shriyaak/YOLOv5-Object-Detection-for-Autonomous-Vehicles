import torch
import os
import numpy as np
from PIL import Image
from datafetch import KittiDataset, collate_fn, CLASS_MAPPING, IGNORE_CLASSES
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import matplotlib.pyplot as plt

# Paths
model_path = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/runs/train/kiti_yolov5/weights/best.pt'
unlabeled_img_dir = "/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/images1/unlabeled"
output_label_dir = "/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/saved"
attention_dir = "/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/attention_maps"

# Creating output directories
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(attention_dir, exist_ok=True)

# Defining transformations
default_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

# Creating dataset for unlabeled images
unlabeled_dataset = KittiDataset(img_dir=unlabeled_img_dir, label_dir=None, transform=default_transform)

def convert_to_yolo_format(img_width, img_height, labels):
    yolo_labels = []
    for label in labels:
        parts = label.split()
        class_name = parts[0]


        if class_name in IGNORE_CLASSES:
            continue

        class_id = CLASS_MAPPING.get(class_name, -1)
        if class_id == -1:
            continue

        x_min = float(parts[4])
        y_min = float(parts[5])
        x_max = float(parts[6])
        y_max = float(parts[7])

        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        yolo_labels.append([class_id, x_center, y_center, width, height])

    return yolo_labels

def save_pseudo_labels(predictions, img_filenames, save_dir, img_width, img_height):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving pseudo-labels to directory: {save_dir}")
    for i, pred in enumerate(predictions):
        img_filename = os.path.splitext(img_filenames[i])[0] + '.txt'
        label_path = os.path.join(save_dir, img_filename)
        print(f"Saving labels to {label_path}")
        try:
            with open(label_path, 'w') as f:
                if not pred:
                    print(f"No predictions for image {img_filenames[i]}")
                for bbox in pred:
                    class_id, x_center, y_center, width, height = bbox
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        except Exception as e:
            print(f"Failed to write {label_path}: {e}")

def get_image_filenames(img_dir):
    return [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

# Generating pseudo-labels
def generate_pseudo_labels(model, dataset, threshold=0.4):
    model.eval()
    pseudo_labels = []
    image_paths = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            img = data['image'].unsqueeze(0)

            # Inference
            preds = model(img)[0]
            img_width, img_height = data['image'].shape[1], data['image'].shape[2]


            boxes = preds[preds[:, 4] > threshold]
            bboxes = []
            for box in boxes:
                class_id = int(box[5].item())
                class_name = list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(class_id)]
                print(f"Predicted class: {class_name}")
                x_min, y_min, x_max, y_max = box[:4].tolist()
                bboxes.append([class_id, x_min, y_min, x_max, y_max])

            # Converting to YOLO format
            if bboxes:
                pseudo_labels.append(convert_to_yolo_format(img_width, img_height, [
                    f"{list(CLASS_MAPPING.keys())[bbox[0]]} 0.0 0 0.0 {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
                    for bbox in bboxes
                ]))
                image_paths.append(dataset.img_filenames[idx])

    return image_paths, pseudo_labels

# Definining dataset for pseudo-labels
class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, pseudo_labels, transform=None):
        self.image_paths = image_paths
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.pseudo_labels[idx], dtype=torch.float32)
        return {'image': image, 'label': label}

def combine_datasets(labeled_loader, image_paths, pseudo_labels, batch_size):
    pseudo_dataset = PseudoLabelDataset(image_paths, pseudo_labels, transform=default_transform)
    combined_dataset = ConcatDataset([labeled_loader.dataset, pseudo_dataset])
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

def visualize_attention_map(image, attention_map, save_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(attention_map, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


image_paths = get_image_filenames(unlabeled_img_dir)


for idx in range(len(image_paths)):
    try:
        img_full_path = os.path.join(unlabeled_img_dir, image_paths[idx])
        if not os.path.exists(img_full_path):
            print(f"Image not found at: {img_full_path}")
            continue

        attention_map = np.random.rand(416, 416)
        attention_map_path = os.path.join(attention_dir, image_paths[idx].replace('.png', '_attention.png'))


        img_pil = Image.open(img_full_path)

        # Visualizing and saving the attention map
        visualize_attention_map(img_pil, attention_map, attention_map_path)
        print(f"Visualization saved to {attention_map_path}")
    except FileNotFoundError as e:
        print(f"Skipping {image_paths[idx]} due to error: {e}")
    except Exception as e:
        print(f"Error during visualization for {image_paths[idx]}: {e}")
else:
    print("No image paths available for visualization.")





