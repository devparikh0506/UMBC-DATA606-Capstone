"""
EEG-DBNet: A Dual-Branch Network for Temporal-Spectral Decoding in Motor-Imagery

Based on:
Lou, X., et al. (2024). EEG-DBNet: A Dual-Branch Network for Temporal-Spectral Decoding 
in Motor-Imagery. arXiv preprint arXiv:2405.16090v3.

Architecture:
- Dual-branch structure: Temporal branch + Spectral branch
- Each branch: Local Convolution (LC) block + Global Convolution (GC) block
- GC block uses sliding window, SE-based feature reconstruction, and DCCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DilatedCausalConv1d(nn.Module):
    """Dilated Causal Convolution 1D layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding, bias=False
        )
    
    def forward(self, x):
        # Apply causal convolution (padding on left only)
        x = self.conv(x)
        # Remove padding from right to maintain causality
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for feature reconstruction."""
    
    def __init__(self, seq_len, reduction=16):
        super().__init__()
        self.seq_len = seq_len
        compressed_len = max(1, (seq_len + reduction - 1) // reduction)  # ceil division
        
        self.fc1 = nn.Linear(seq_len, compressed_len)
        self.fc2 = nn.Linear(compressed_len, seq_len)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        # Global average pooling across channel dimension
        pooled = x.mean(dim=1)  # (batch, seq_len)
        # Squeeze and excite
        compressed = F.relu(self.fc1(pooled))  # (batch, compressed_len)
        weights = torch.sigmoid(self.fc2(compressed))  # (batch, seq_len)
        # Apply weights to each channel
        weights = weights.unsqueeze(1)  # (batch, 1, seq_len)
        return x * weights  # Hadamard product


class LocalConvolutionBlock(nn.Module):
    """Local Convolution Block as described in EEG-DBNet paper.
    
    Similar to EEGNet structure:
    - First conv: temporal convolution
    - Depthwise: spatial convolution
    - Separable: depthwise temporal + pointwise
    """
    
    def __init__(self, num_channels, fs=250, F1=8, D=2, K=48, 
                 pool_type='avg', dropout_rate=0.3):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.K = K
        self.F2 = F1 * D
        
        # First convolutional layer (temporal)
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, K),
                     padding=(0, K // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # Depthwise spatial convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1),
                     groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
        )
        
        # Pooling layer
        pool_size = K // 8
        if pool_type == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=(1, pool_size))
        else:  # max pooling
            self.pool1 = nn.MaxPool2d(kernel_size=(1, pool_size))
        
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Separable convolution
        sep_kernel = K // 4
        self.separable = nn.Sequential(
            # Depthwise temporal
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, sep_kernel),
                     padding=(0, sep_kernel // 2), groups=F1 * D, bias=False),
            # Pointwise
            nn.Conv2d(F1 * D, self.F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
        )
        
        # Second pooling
        if pool_type == 'avg':
            self.pool2 = nn.AvgPool2d(kernel_size=(1, pool_size))
        else:  # max pooling
            self.pool2 = nn.MaxPool2d(kernel_size=(1, pool_size))
        
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x shape: (batch, 1, channels, time)
        x = self.firstconv(x)
        x = self.depthwise(x)  # (batch, F1*D, 1, time)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.separable(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        # Remove spatial dimension (should be 1 after depthwise)
        x = x.squeeze(2)  # (batch, F2, time)
        return x


class GlobalConvolutionBlock(nn.Module):
    """Global Convolution Block with sliding window, SE, and DCCNN."""
    
    def __init__(self, in_channels, seq_len, n_windows=6, n_dcc_layers=4, 
                 dcc_kernel_size=4, dropout_rate=0.3):
        super().__init__()
        self.n_windows = n_windows
        self.seq_len = seq_len
        self.window_len = seq_len - n_windows + 1  # stride=1
        
        # SE blocks for each subsequence
        self.se_blocks = nn.ModuleList([
            SEBlock(self.window_len) for _ in range(n_windows)
        ])
        
        # DCCNN layers for each subsequence
        # Paper: dilation rate r = j for j-th DCC layer
        self.dcc_layers = nn.ModuleList()
        for _ in range(n_windows):
            window_dcc_layers = nn.ModuleList()
            for j in range(n_dcc_layers):
                dilation = j + 1  # dilation rates: 1, 2, 3, 4 for layers 0, 1, 2, 3
                window_dcc_layers.append(
                    DilatedCausalConv1d(in_channels, in_channels, 
                                       dcc_kernel_size, dilation=dilation)
                )
            self.dcc_layers.append(window_dcc_layers)
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(in_channels) for _ in range(n_windows)
        ])
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        batch_size, channels, seq_len = x.shape
        
        # Create sliding windows (stride=1)
        subsequences = []
        for i in range(self.n_windows):
            start = i
            end = start + self.window_len
            subseq = x[:, :, start:end]  # (batch, channels, window_len)
            subsequences.append(subseq)
        
        # Process each subsequence
        processed = []
        for i, subseq in enumerate(subsequences):
            # SE-based feature reconstruction
            subseq_se = self.se_blocks[i](subseq)
            
            # DCCNN with residual connections (paper: output added to source before next layer)
            residual = subseq_se
            for dcc_layer in self.dcc_layers[i]:
                subseq_out = dcc_layer(residual)
                subseq_out = F.elu(subseq_out)
                # Residual connection: add output to source subsequence before next layer
                residual = subseq_se + subseq_out
            
            # Batch normalization
            residual = self.batch_norms[i](residual)
            processed.append(residual)
        
        # Concatenate all processed subsequences
        output = torch.cat(processed, dim=2)  # (batch, channels, n_windows * window_len)
        return output


class EEG_DBNet(nn.Module):
    """EEG-DBNet: Dual-Branch Network for Motor-Imagery EEG Classification.
    
    Architecture:
    - Temporal branch: LC block (avg pooling) + GC block
    - Spectral branch: LC block (max pooling) + GC block
    - Concatenate outputs and classify with FC layer
    
    Args:
        num_classes: Number of classes (2 for BCI IV-2b)
        num_channels: Number of EEG channels (3 for BCI IV-2b: Cz, C3, C4)
        fs: Sampling frequency (250 Hz for BCI IV-2b)
        T: Number of time samples (1000 for 4 seconds at 250Hz, 
           1125 for 4.5 seconds at 250Hz for [2.5, 7] window)
        dropout_rate: Dropout rate (0.3 for LC blocks)
    """
    
    def __init__(self, num_classes=2, num_channels=3, fs=250, T=1125,
                 dropout_rate=0.3):
        super().__init__()
        
        # Temporal branch parameters
        F1_temporal = 8
        K_temporal = 48  # K̂
        D = 2
        F_temporal = F1_temporal * D  # 16
        
        # Spectral branch parameters
        F1_spectral = 16
        K_spectral = 64  # K̃
        F_spectral = F1_spectral * D  # 32
        
        # GC block parameters
        n_windows = 6
        n_dcc_layers = 4
        dcc_kernel_size = 4
        
        # Temporal branch LC block (average pooling)
        self.temporal_lc = LocalConvolutionBlock(
            num_channels=num_channels,
            fs=fs,
            F1=F1_temporal,
            D=D,
            K=K_temporal,
            pool_type='avg',
            dropout_rate=dropout_rate
        )
        
        # Calculate temporal branch output length after LC
        # T̂ = floor(64 * T / K̂²)
        T_temporal = (64 * T) // (K_temporal ** 2)
        
        # Temporal branch GC block
        self.temporal_gc = GlobalConvolutionBlock(
            in_channels=F_temporal,
            seq_len=T_temporal,
            n_windows=n_windows,
            n_dcc_layers=n_dcc_layers,
            dcc_kernel_size=dcc_kernel_size,
            dropout_rate=dropout_rate
        )
        
        # Spectral branch LC block (max pooling)
        self.spectral_lc = LocalConvolutionBlock(
            num_channels=num_channels,
            fs=fs,
            F1=F1_spectral,
            D=D,
            K=K_spectral,
            pool_type='max',
            dropout_rate=dropout_rate
        )
        
        # Calculate spectral branch output length after LC
        # T̃ = floor(64 * T / K̃²)
        T_spectral = (64 * T) // (K_spectral ** 2)
        
        # Spectral branch GC block
        self.spectral_gc = GlobalConvolutionBlock(
            in_channels=F_spectral,
            seq_len=T_spectral,
            n_windows=n_windows,
            n_dcc_layers=n_dcc_layers,
            dcc_kernel_size=dcc_kernel_size,
            dropout_rate=dropout_rate
        )
        
        # Calculate output dimensions
        temporal_window_len = T_temporal - n_windows + 1
        temporal_output_len = n_windows * temporal_window_len
        
        spectral_window_len = T_spectral - n_windows + 1
        spectral_output_len = n_windows * spectral_window_len
        
        # Flatten and concatenate
        self.fc = nn.Linear(
            F_temporal * temporal_output_len + F_spectral * spectral_output_len,
            num_classes
        )
    
    def forward(self, x):
        # x shape: (batch, channels, time) or (batch, 1, channels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, channels, time)
        
        # Temporal branch
        temporal = self.temporal_lc(x)  # (batch, F_temporal, T_temporal)
        temporal = self.temporal_gc(temporal)  # (batch, F_temporal, expanded_len)
        temporal = temporal.flatten(1)  # (batch, F_temporal * expanded_len)
        
        # Spectral branch
        spectral = self.spectral_lc(x)  # (batch, F_spectral, T_spectral)
        spectral = self.spectral_gc(spectral)  # (batch, F_spectral, expanded_len)
        spectral = spectral.flatten(1)  # (batch, F_spectral * expanded_len)
        
        # Concatenate
        combined = torch.cat([temporal, spectral], dim=1)
        
        # Classification
        output = self.fc(combined)
        return output

