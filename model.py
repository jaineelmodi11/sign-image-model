# model.py — from-scratch CNN (no pretrained)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → (optional Dropout2d) → MaxPool(2)"""
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.drop(x)
        x = self.pool(x)
        return x


class SimpleSignNet(nn.Module):
    """
    Input: 3x224x224
    Features: 32 -> 64 -> 128 -> 256 (4 blocks, each halves H/W)
    Head: GAP -> 256 -> 128 -> num_classes
    """
    def __init__(self, num_classes: int = 29, drop_p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32, 0.05),
            ConvBlock(32,  64, 0.10),
            ConvBlock(64, 128, 0.15),
            ConvBlock(128, 256, 0.20),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B,256,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),            # (B,256)
            nn.Dropout(drop_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(128, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
