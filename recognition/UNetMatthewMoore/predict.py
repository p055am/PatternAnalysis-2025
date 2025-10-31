#Shows example usage of the trained model. Print out any results and/or provide visualisations where applicable

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


from modules import SimpleUNet
import dataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Predict Using device: {device}')

def evaluate_model(model_path, test_dataset, num_classes=6):
    """
    Loads a saved model and evaluates Dice scores for each class on the given dataset.

    Args:
        model_path (str): Path to the saved model .pth file
        dataset (torch.utils.data.Dataset): Dataset to evaluate (e.g., test set)
        num_classes (int): Number of segmentation classes (default 6)
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import numpy as np


    # --- Load model ---
    model = SimpleUNet(in_channels=1, out_channels=num_classes, dropout_p=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Sets the model to evaluation mode

    # --- Initialize Dice tracking ---
    dice_scores = np.zeros(num_classes)
    eps = 1e-6  # avoid divide-by-zero

    # --- Iterate through dataset ---
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Evaluating model on dataset...")
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images, masks = images.to(device), masks.to(device)

            preds = model(images)  # [B, C, H, W]
            preds = F.softmax(preds, dim=1)
            
            # Compute Dice per class
            preds_flat = preds.view(preds.shape[0], preds.shape[1], -1)
            masks_flat = masks.view(masks.shape[0], masks.shape[1], -1)

            intersection = (preds_flat * masks_flat).sum(dim=2)
            union = preds_flat.sum(dim=2) + masks_flat.sum(dim=2)

            dice = (2.0 * intersection + eps) / (union + eps)
            dice_scores += dice.mean(dim=0).cpu().numpy()  # average across batch

    # Average across dataset
    dice_scores /= len(loader)

    # --- Report results ---
    segment_labels = ['Empty', 'Body Outline', 'Bone', 'Bladder', 'Rectum', 'Prostate'][:num_classes]
    print("Dice Scores per Class:")
    for label, score in zip(segment_labels, dice_scores):
        print(f"  {label:<12}: {score:.4f}")

    print(f"Mean Dice Score: {dice_scores.mean():.4f}")
    return dice_scores


# Runs the evaluation
print("Getting testing data")
test_img_path = dataset.get_image_path_hip_mri('test', 'image')
test_mask_path = dataset.get_image_path_hip_mri('test', 'mask')
test_dataset = dataset.DataSegmenter2D(test_img_path, test_mask_path)
print(f"Number of test samples: {len(test_dataset)}")

evaluate_model('unet_hip_mri.pth', test_dataset)