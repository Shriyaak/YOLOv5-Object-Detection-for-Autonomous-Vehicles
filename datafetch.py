import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

# SSL context setup
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Updated Class mappings for KITTI dataset
CLASS_MAPPING = {
    'Car': 0,
    'Truck': 1,
    'Van': 2,
    'Misc': 3
}

IGNORE_CLASSES = ['DontCare', 'Pedestrian', 'Bus', 'Person_sitting', 'Cyclist', 'Tram']

class KittiDataset(Dataset):
    def __init__(self, img_dir, label_dir=None, transform=None, target_size=(416, 416)):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size


        self.img_filenames = sorted([os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')])
        print(f"Found {len(self.img_filenames)} images in {self.img_dir}")

        # Check if it's a labeled dataset
        if label_dir:
            # List label filenames
            label_filenames = sorted([os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')])
            print(f"Found {len(label_filenames)} labels in {self.label_dir}")

            # Filter files based on common filenames (only if both labels and images are available)
            img_prefixes = set(self.img_filenames)
            label_prefixes = set(label_filenames)
            common_prefixes = img_prefixes & label_prefixes

            # keeping common filenames if the label directory is provided
            self.img_filenames = [f for f in self.img_filenames if f in common_prefixes]
            print(f"Filtered {len(self.img_filenames)} common filenames")
        else:
            self.label_filenames = None  # No labels for unlabeled dataset

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename + '.png')

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return {'image': None, 'label': torch.empty((0, 5), dtype=torch.float32)}

        # image resizing
        image = image.resize(self.target_size, Image.BILINEAR)

        # image conversion to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label_data = []
        if self.label_dir:
            label_path = os.path.join(self.label_dir, img_filename + '.txt')
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as file:
                        for line in file:
                            label = self.parse_label(line)
                            if label:
                                label_data.append(label)
                except Exception as e:
                    print(f"Error reading label file {label_path}: {e}")

        # converting labelled data to tensor
        if label_data:
            label_data = torch.tensor(label_data, dtype=torch.float32)
        else:
            label_data = torch.empty((0, 5), dtype=torch.float32)  # Empty tensor for no labels

        return {'image': image, 'label': label_data}

    def parse_label(self, label_line):
        elements = label_line.strip().split()
        if len(elements) != 5:
            return None

        try:
            class_id = int(float(elements[0]))
            x_center = float(elements[1])
            y_center = float(elements[2])
            width = float(elements[3])
            height = float(elements[4])
        except ValueError:
            return None

        if class_id in CLASS_MAPPING.values():  # Ensure class_id is valid
            return [class_id, x_center, y_center, width, height]
        return None

# custom Collate_fn for mean teacher
def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]

    # Padding labels to have consistent size
    max_boxes = max([label.size(0) for label in labels])
    padded_labels = []
    for label in labels:
        if label.size(0) < max_boxes:
            pad_size = max_boxes - label.size(0)
            padded_label = torch.cat([label, torch.zeros(pad_size, label.size(1))], dim=0)
        else:
            padded_label = label
        padded_labels.append(padded_label)

    images = torch.stack(images, 0)
    labels = torch.stack(padded_labels, 0)

    return {'image': images, 'label': labels}




