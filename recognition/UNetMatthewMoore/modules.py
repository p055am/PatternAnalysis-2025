# Source code of the components of the model.
# Each component must be implemented as a class or function.

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

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class SimpleUNet(nn.Module):
    """Simplified U-Net for demo purposes with batch normalization, LeakyReLU, dropout and sigmoid activation."""

    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.2):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, 32, dropout_p)
        self.enc2 = self._conv_block(32, 64, dropout_p)
        self.enc3 = self._conv_block(64, 128, dropout_p)

        # Decoder (upsampling)
        # The +64 and +32 are to account fo the skip connections from the encoder
        self.dec3 = self._conv_block(128 + 64, 64, dropout_p)
        self.dec2 = self._conv_block(64 + 32, 32, dropout_p)
        self.dec1 = nn.Conv2d(32, out_channels, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for final output

    def _conv_block(self, in_ch, out_ch, dropout_p=0.2):
        """Conv block with batch normalization and LeakyReLU:
        Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU -> Dropout"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_p)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x) # 64x64
        e2 = self.enc2(self.pool(e1))  # 32x32
        e3 = self.enc3(self.pool(e2))  # 16x16

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(e3), e2], 1))  # 32x32
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], 1))  # 64x64
        out = self.dec1(d2)

        # Apply sigmoid activation to final output
        out = self.sigmoid(out)

        return out