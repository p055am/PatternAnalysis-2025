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
from matplotlib.colors import ListedColormap
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
            # Apparantly, non-floating tensors can't be moved to CUDA
            images, masks = images.float().to(device), masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # print the shape of outputs and masks for debugging
            # print(f"Predicted shape: {outputs.shape}, masks shape: {masks.shape}")
            loss = criterion(outputs, masks)

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
    """Multi-class Dice Loss function

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
            predictions: Raw model outputs or propabilities [B, C, H, W]
            targets: One-hot encoded ground truth masks [B, C, H, W]
        """
        # If predictions are not probabilities, apply softmax across classes
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")

        # Convert logits to probabilities (if not already)
        if not torch.all((predictions >= 0) & (predictions <= 1)):
            predictions = F.softmax(predictions, dim=1)

        # Flatten each class for batchwise computation
        predictions = predictions.contiguous().view(predictions.shape[0], predictions.shape[1], -1)
        targets = targets.contiguous().view(targets.shape[0], targets.shape[1], -1).float()

        # Compute intersection and union per class
        intersection = (predictions * targets).sum(dim=2)
        dice_score = (2.0 * intersection + self.smooth) / (
            predictions.sum(dim=2) + targets.sum(dim=2) + self.smooth
        )

        # Average across classes and batch
        dice_loss = 1 - dice_score.mean()

        return dice_loss
    

#create visualization functions

def denormalize_image(tensor):
    """Denormalize a tensor image with ImageNet mean and std."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_tensor = tensor * std + mean
    return torch.clamp(denorm_tensor, 0, 1)

def show_examples(dataset, title="Dataset Examples", n=5, num_classes=6, starting_index=0):
    """Quick visualization of some Hip MRI data, with labelled masks."""
    fig, axes = plt.subplots(2, n, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Define a color map for masks (one color per class)
    segment_labels = ['Empty', 'Body Outline', 'Bone', 'Bladder', 'Rectum', 'Prostate'][:num_classes]
    segment_colors = ['black', 'red', 'green', 'blue', 'yellow', 'magenta'][:num_classes]
    cmap = ListedColormap(segment_colors) # type: ignore

    for i in range(n):
        index = i + starting_index
        image, mask = dataset[index] # mask is one-hot: [C, H, W]. 

        # Denormalize image for visualization
        img_show = denormalize_image(image)

        # Show image (transpose from CHW to HWC for matplotlib)
        img_display = img_show.permute(1, 2, 0).numpy()  # CHW -> HWC
        axes[0, i].imshow(img_display, cmap='gray')
        axes[0, i].set_title(f'Hip Image {index+1}', fontweight='bold')
        axes[0, i].axis('off')

        # Convert one-hot mask to a single integer mask for plotting
        # mask: [C,H,W] -> [H,W] with values 0..num_classes-1
        mask_np = torch.argmax(mask, dim=0).numpy()

        im = axes[1, i].imshow(mask_np, cmap=cmap, vmin=0, vmax=num_classes-1)
        axes[1, i].set_title(f'Mask {index+1}', fontweight='bold')
        axes[1, i].axis('off')

        # Add colorbar for the first image to show the scale
        if i == 0:
            cbar = plt.colorbar(im, ax=axes[1, i], shrink=0.6, ticks=range(num_classes))
            cbar.ax.set_yticklabels(segment_labels)
            cbar.set_label('Hip MRI Segments', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

def show_epoch_predictions(model, dataset, epoch, n=3, num_classes=6):
    """Show model predictions after a specific epoch."""
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(f'🎯 Predictions After Epoch {epoch}', fontsize=16, fontweight='bold')

    # Segment labels and colors (same as show_examples)
    segment_labels = ['Empty', 'Body Outline', 'Bone', 'Bladder', 'Rectum', 'Prostate'][:num_classes]
    segment_colors = ['black', 'orange', 'blue', 'green', 'red', 'magenta'][:num_classes]
    cmap = ListedColormap(segment_colors)

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]

            # Predict with sigmoid model
            pred = model(image.unsqueeze(0).to(device)) # The unsqueeze makes sure it has batch dimension
            pred_mask = torch.argmax(pred.squeeze(0), dim=0).cpu().numpy() # [H, W]

            # Ground truth: convert one-hot to class indices
            true_mask_np = torch.argmax(true_mask, dim=0).numpy()

            # Denormalize image for visualization
            img_show = denormalize_image(image)

            # Show original color image (transpose from CHW to HWC for matplotlib)
            img_display = img_show.permute(1, 2, 0).numpy()  # CHW -> HWC
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')

            # Show ground truth mask
            axes[1, i].imshow(true_mask_np, cmap=cmap, vmin=0, vmax=num_classes)
            axes[1, i].set_title(f'Ground Truth {i+1}', fontweight='bold')
            axes[1, i].axis('off')

            # Show prediction with accuracy
            axes[2, i].imshow(pred_mask, cmap=cmap, vmin=0, vmax=num_classes)
            accuracy = np.mean(pred_mask == true_mask_np)
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

# Create datasets with subset for faster training/demo
# Set subset_size=None to use full dataset, or specify a number for quick demo
subset_size = 500  # Use 100 samples for demo
# subset_size = None  # Uncomment this to use full dataset

print("🔄 Loading datasets (normalized to zero mean & unit std)...")
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

# Show examples
show_examples(train_dataset, "Initial examples", starting_index=30)

# 1 input channel because grayscale, 6 output channels because 6 segments
model = modules.SimpleUNet(in_channels=1, out_channels=6, dropout_p=0.2)
losses = train(model, train_loader, test_dataset, epochs=21, lr=0.001, visualize_every=10)
plot_loss(losses, loss_type='dice')