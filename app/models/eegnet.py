"""
EEGNet: Compact Convolutional Neural Network for EEG Classification

Implementation based on Lawhern et al. (2018).
Paper: EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet: Compact Convolutional Neural Network for EEG Classification.
    
    Architecture (Lawhern et al., 2018):
    - Block 1: Temporal Conv → Depthwise Conv → Separable Conv → AvgPool
    - Block 2: Depthwise Conv → Separable Conv → AvgPool
    - Classification: Dense layer with dropout
    
    Args:
        num_channels: Number of EEG channels (3 for BCI IV-2b)
        num_classes: Number of classes (2 for left/right hand)
        F1: Number of temporal filters (default 8)
        F2: Number of pointwise filters (default 16)
        D: Number of spatial filters per temporal filter (default 2)
        kernel_length: Length of temporal convolution kernel (default 64)
        pool_time: Pooling size in time dimension (default 4)
        pool_space: Pooling size in space dimension (default 1)
        dropout_rate: Dropout rate (default 0.5)
    """
    
    def __init__(self, 
                 num_channels=3,
                 num_classes=2,
                 F1=8,
                 F2=16,
                 D=2,
                 kernel_length=64,
                 pool_time=4,
                 pool_space=1,
                 dropout_rate=0.5):
        super().__init__()
        
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), 
                                       padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.conv_spatial = nn.Conv2d(F1, D * F1, (num_channels, 1), 
                                      groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((pool_space, pool_time))
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.conv_separable_depth = nn.Conv2d(D * F1, D * F1, (1, 16),
                                               groups=D * F1, padding=(0, 8), bias=False)
        self.conv_separable_point = nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, num_classes, bias=True)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            output: Class logits of shape (batch, num_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv_temporal(x)
        x = self.bn1(x)
        
        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        output = self.classifier(x)
        return output

