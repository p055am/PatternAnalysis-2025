# Contains the source code for training validating, testing and saving
# the model. The model should be imported from modules.py and the
# data loader should be imported from dataset.py

# Make sure to plot the losses and metrics during training

# Reference: This code is modified from the UNet segmentations task
# on blackboard. Link: 
# https://colab.research.google.com/drive/1VOsZSyRhyuHLmgoqGriQk01ub4bKNmZ1?usp=sharing#scrollTo=0ec83094

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import random
import modules
import dataset

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def train(model, train_loader, test_dataset, epochs=3, lr=0.001, visualize_every=1):
    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(" Starting training with Batch Norm, LeakyReLU, and Sigmoid activation...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop with progress
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            pred_pet = outputs[:, 0]  # Pet class probability from sigmoid
            #print the shape of pred_pet and masks for debugging
            # print(f"pred_pet shape: {outputs.shape}, masks shape: {masks.shape}")
            loss = criterion(pred_pet, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        # Visualize predictions after each epoch (or every few epochs)
        if (epoch) % visualize_every == 0:
            show_epoch_predictions(model, test_dataset, epoch + 1, n=3)

    print(" Training complete with enhanced U-Net!")
    return losses


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Sigmoid output from model [B, H, W] (values between 0-1)
            targets: Binary ground truth [B, H, W] (values 0 or 1)
        """
        # Flatten tensors using reshape to handle non-contiguous memory layout
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - dice_coeff
    

#create visualization functions

def denormalize_image(tensor):
    """Denormalize a tensor image with ImageNet mean and std."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_tensor = tensor * std + mean
    return torch.clamp(denorm_tensor, 0, 1)

def show_examples(dataset, title="Dataset Examples", n=3):
    """Quick visualization for color demo with binary masks."""
    fig, axes = plt.subplots(2, n, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i in range(n):
        image, mask = dataset[i]

        # Denormalize image for visualization
        img_show = denormalize_image(image)

        # Show color image (transpose from CHW to HWC for matplotlib)
        img_display = img_show.permute(1, 2, 0).numpy()  # CHW -> HWC
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f'Pet Image {i+1} (Color RGB)', fontweight='bold')
        axes[0, i].axis('off')

        # Debug mask values for this sample
        mask_np = mask.numpy()
        unique_vals = np.unique(mask_np)
        pet_count = np.sum(mask_np == 1)
        bg_count = np.sum(mask_np == 0)

        # Show binary mask with better colormap
        im = axes[1, i].imshow(mask_np, cmap='RdBu', vmin=0, vmax=1)
        axes[1, i].set_title(f'Mask {i+1} (Pet:{pet_count}, BG:{bg_count})', fontweight='bold')
        axes[1, i].axis('off')

        # Add colorbar for the first image to show the scale
        if i == 0:
            from matplotlib.colors import ListedColormap
            colors = ['blue', 'red']  # blue for background (0), red for pet (1)
            cmap = ListedColormap(colors)
            im = axes[1, i].imshow(mask_np, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[1, i], shrink=0.6, ticks=[0, 1], label='0=BG, 1=Pet')

    plt.tight_layout()
    plt.show()

def show_epoch_predictions(model, dataset, epoch, n=3):
    """Show model predictions after a specific epoch."""
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(f'🎯 Predictions After Epoch {epoch}', fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]

            # Predict with sigmoid model
            pred = model(image.unsqueeze(0).to(device))
            # Get pet class probability and convert to binary
            pred_pet_prob = pred[0, 0].cpu().numpy()  # Pet class probability
            pred_binary = (pred_pet_prob > 0.5).astype(int)  # Binary prediction

            # Denormalize image for visualization
            img_show = denormalize_image(image)

            # Show original color image (transpose from CHW to HWC for matplotlib)
            img_display = img_show.permute(1, 2, 0).numpy()  # CHW -> HWC
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')

            # Show ground truth binary mask
            axes[1, i].imshow(true_mask, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[1, i].set_title(f'Ground Truth {i+1}', fontweight='bold')
            axes[1, i].axis('off')

            # Show prediction with accuracy
            axes[2, i].imshow(pred_binary, cmap='RdYlBu_r', vmin=0, vmax=1)
            accuracy = np.mean(pred_binary == true_mask.numpy())
            axes[2, i].set_title(f'Prediction {i+1} (Acc: {accuracy:.3f})', fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
    model.train()  # Switch back to training mode

# Quick visualization of loss
def plot_loss(losses, loss_type='dice'):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'bo-', linewidth=2, markersize=8)

    title_map = {
        'bce': '🔥 Training Loss (BCE)',
        'dice': '🔥 Training Loss (Dice)',
        'combined': '🔥 Training Loss (Combined BCE + Dice)'
    }
    plt.title(title_map.get(loss_type, '🔥 Training Loss'), fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()



# Transforms for color demo with normalization to zero mean and unit std
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),  # This converts to 0-1 range automatically for RGB
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet RGB means
        std=[0.229, 0.224, 0.225]    # ImageNet RGB standard deviations
    )
])

# Create datasets with subset for faster training/demo
# Set subset_size=None to use full dataset, or specify a number for quick demo
subset_size = 1000  # Use 1000 samples for demo
# subset_size = None  # Uncomment this to use full dataset

print("🔄 Loading datasets (color RGB, normalized to zero mean & unit std)...")
# Get file lists
print("Getting training data")
train_img_path = dataset.get_image_path_hip_mri('train', 'image')
train_mask_path = dataset.get_image_path_hip_mri('train', 'mask')
train_dataset = dataset.DataSegmenter2D(train_img_path, train_mask_path, subset_size=subset_size)
print(f"Number of training samples: {len(train_dataset)}")

print("Getting validation data")
validate_img_path = dataset.get_image_path_hip_mri('validate', 'image')
validate_mask_path = dataset.get_image_path_hip_mri('validate', 'mask')
validate_dataset = dataset.DataSegmenter2D(validate_img_path, validate_mask_path, subset_size=(subset_size // 2))
print(f"Number of validation samples: {len(validate_dataset)}")

print("Getting testing data")
test_img_path = dataset.get_image_path_hip_mri('test', 'image')
test_mask_path = dataset.get_image_path_hip_mri('test', 'mask')
test_dataset = dataset.DataSegmenter2D(test_img_path, test_mask_path, subset_size=(subset_size // 2))
print(f"Number of test samples: {len(test_dataset)}")

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)
print("✅ Data loaders ready! (Color RGB + Binary masks + Normalized)")
print("✅ Data loaders ready! (Color RGB + Binary masks + 0-1 normalization)")

# Show examples
show_examples(train_dataset, "🐕 Color Pet Dataset + Binary Masks (Normalized)")

model = modules.SimpleUNet(in_channels=3, out_channels=1, dropout_p=0.2)
losses = train(model, train_loader, test_dataset, epochs=1000, lr=0.001, visualize_every=50)
plot_loss(losses, loss_type='dice')