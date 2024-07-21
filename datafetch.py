
#os.listdir('./kitti_dataset/val/val_label ')

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class KittiDataset(Dataset):
    def __init__(self, img_dir, label_dir=None, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # List image filenames (without extensions)
        self.img_filenames = sorted([os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')])

        if label_dir:
            # List label filenames (without extensions)
            label_filenames = sorted([os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')])

            # Extract prefixes (first 6 characters)
            img_prefixes = {f[:] for f in self.img_filenames}
            label_prefixes = {f[:] for f in label_filenames}

            # Find common prefixes
            common_prefixes = img_prefixes & label_prefixes
            # Filter files based on common prefixes
            self.img_filenames = [f for f in self.img_filenames if f[:] in common_prefixes]
            self.label_filenames = [f for f in label_filenames if f[:] in common_prefixes]

            # Check that filenames match
            if set([os.path.splitext(f)[0] for f in self.img_filenames]) != set(
                    [os.path.splitext(f)[0] for f in self.label_filenames]):
                raise ValueError("Mismatch between image and label filenames after filtering.")

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename + '.png')

        # Load image
        image = Image.open(img_path).convert("RGB")

        label = None
        if self.label_dir:
            label_path = os.path.join(self.label_dir, img_filename + '.txt')
            # Load label (as text)
            with open(label_path, 'r') as file:
                label = file.read().strip()

        if self.transform:
            image = self.transform(image)

        # Return a dictionary with the image and optionally the label
        if label is not None:
            return {'image': image, 'label': label}
        else:
            return {'image': image}

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Correct paths to your dataset
train_img_dir = './kitti_dataset/train/train_img'
train_label_dir = './kitti_dataset/train/train_label'
val_img_dir = './kitti_dataset/val/val_img'
val_label_dir = './kitti_dataset/val/val_label'
test_img_dir = './kitti_dataset/test/test_img'

# Get the absolute paths
train_img_dir_abs = os.path.abspath(train_img_dir)
train_label_dir_abs = os.path.abspath(train_label_dir)
val_img_dir_abs = os.path.abspath(val_img_dir)
val_label_dir_abs = os.path.abspath(val_label_dir)
test_img_dir_abs = os.path.abspath(test_img_dir)

# Verify directories
for dir_path, name in [(train_img_dir_abs, "training image"),
                       (train_label_dir_abs, "training label"),
                       (val_img_dir_abs, "validation image"),
                       (val_label_dir_abs, "validation label"),
                       (test_img_dir_abs, "testing image")]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"The {name} directory {dir_path} does not exist.")
    else:
        print(f"{name.capitalize()} directory contents:", os.listdir(dir_path))

# Instantiate datasets
train_dataset = KittiDataset(img_dir=train_img_dir,
                             label_dir=train_label_dir,
                             transform=transform)

val_dataset = KittiDataset(img_dir=val_img_dir,
                           label_dir=val_label_dir,
                           transform=transform)

test_dataset = KittiDataset(img_dir=test_img_dir,
                            transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)









