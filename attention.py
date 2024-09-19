import torch
import matplotlib.pyplot as plt
import numpy as np
from CR import *  # Import everything from CR.py if needed


def visualize_attention(image, attention_map, title='Attention Map'):
    #converting tensor to numpy array
    image = image.permute(1, 2, 0).cpu().numpy()
    attention_map = attention_map.squeeze().cpu().numpy()

    # Normalize attention map to [0, 1] for visualization
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Ploting the image and attention map
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(image)
    plt.imshow(attention_map, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

def visualize_attention_from_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch['image'], batch['label']
            images = images.to(device)

            # Forward pass to get attention maps
            attention_maps = model(images)

            for img, att_map in zip(images, attention_maps):
                visualize_attention(img, att_map)
                break


if __name__ == "__main__":

    from datafetch import KittiDataset, collate_fn
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch

    # Paths to data
    val_img_dir = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/images1/val'
    val_lbl_dir = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/kittii/images1/val_labels'

    # Transformations
    default_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initializing dataset and dataloader
    val_dataset = KittiDataset(val_img_dir, val_lbl_dir, transform=default_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Loading model
    model_path = '/Users/shriyakumbhoje/Desktop/dissertation/pythonProject/yolov5/runs/train/fine_tunned/weights/best.pt'
    model = torch.load(model_path)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Visualizing attention maps
    visualize_attention_from_model(model, val_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
