import datafetch
from datafetch import KittiDataset, train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define transformations for training data (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(10),      # Randomly rotate the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

# Define transformations for validation and test data (no augmentation, just normalization)
val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

# Update datasets with the new transformations
train_dataset = KittiDataset(img_dir=train_img_dir,
                             label_dir=train_label_dir,
                             transform=train_transform)

val_dataset = KittiDataset(img_dir=val_img_dir,
                           label_dir=val_label_dir,
                           transform=val_test_transform)

test_dataset = KittiDataset(img_dir=test_img_dir,
                            transform=val_test_transform)

# Create DataLoaders with the updated datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Just for debugging, print the sizes of the datasets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

