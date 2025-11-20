# Model Training Documentation

This document describes the training strategies and configurations used for training EEGNet models on the BCI Competition IV 2b dataset.

## Overview

The training process compares two strategies for motor imagery classification:
1. **Subject-Dependent Training**: Individual models per subject using cross-validation
2. **Subject-Independent Training**: Leave-One-Subject-Out (LOSO) cross-validation

## Model Architecture: EEG-DBNet

The EEG-DBNet architecture is a dual-branch network for temporal-spectral decoding in motor-imagery EEG classification, based on Lou et al. (2024).

### Architecture Details
- **Input**: 3 channels (Cz, C3, C4) × 1125 time points (4.5 seconds at 250 Hz, [2.5, 7] window)
- **Dual-Branch Structure**: 
  - **Temporal Branch**: LC block (avg pooling) + GC block
  - **Spectral Branch**: LC block (max pooling) + GC block
- **Local Convolution (LC) Blocks**: Similar to EEGNet structure (temporal conv → depthwise conv → separable conv)
- **Global Convolution (GC) Blocks**: Sliding window, SE-based feature reconstruction, DCCNN
- **Output**: 2 classes (Left hand vs Right hand motor imagery)

### Hyperparameters (from paper)
- **Temporal branch**: F1=8, K̂=48, F̂=16, average pooling
- **Spectral branch**: F̃1=16, K̃=64, F̃=32, max pooling
- **GC block**: s=1 (stride), n=6 (windows), d=4 (DCC layers), k=4 (kernel size)
- **Dropout Rate**: 0.3 (in LC blocks)

## Training Configuration

### Common Settings (from paper)
- **Optimizer**: Adam
- **Learning Rate**: 0.0009 (paper uses 0.0009)
- **Weight Decay**: 0.0 (paper doesn't mention weight decay)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Max Epochs**: 1000 (paper uses 1000)
- **Early Stopping**: Patience of 300 epochs, monitors validation loss
- **Training Rounds**: 10 rounds, select best test result

### Subject-Dependent Training
- **Cross-Validation**: 3-fold StratifiedShuffleSplit
- **Validation Size**: 30% of subject's data
- **Data Strategy**: All sessions per subject are combined before CV splitting
- **Model Storage**: `app/resources/models/{subject_id}/best_model.pth`

### Subject-Independent Training
- **Strategy**: Leave-One-Subject-Out (LOSO)
- **Training Data**: Exactly 3 trials (1 from each of 3 randomly selected subjects)
- **Test Data**: All data from the held-out subject
- **Train/Val Split**: 80/20 split of the 3-subject training data
- **Model Storage**: `app/resources/models/{test_subject}/best_model.pth`

## Data Preprocessing

- **Filtering**: 4-40 Hz bandpass filter, 50 Hz notch filter
- **Epochs**: 2.5-7.0 seconds relative to motor imagery cue (paper specification for BCI IV-2b)
  - This extracts 4.5 seconds (1125 samples at 250 Hz)
- **Signal Standardization**: Per-electrode z-score normalization (paper Equations 1-2)
  - Compute mean and std for each electrode across all training trials
  - Apply training statistics to standardize both training and test data
  - Ensures proper cross-validation (test data uses training distribution)
- **Channels**: Cz, C3, C4 (motor cortex channels)
- **Baseline Correction**: None
- **Trial Rejection**: Automatically excludes rejected trials (event code 1023)

## Training Process

1. **Data Loading**: Load and combine all sessions for each subject
2. **Cross-Validation Split**: Stratified split ensuring class balance
3. **Model Initialization**: Fresh model for each fold/subject
4. **Training Loop**: 
   - Forward pass, loss calculation, backpropagation
   - Validation after each epoch
   - Learning rate reduction on plateau
   - Early stopping if no improvement
5. **Model Selection**: Best model based on validation loss
6. **Evaluation**: Accuracy on validation/test set

## Results Storage

- **Training Logs**: `notebooks/outputs/temp_*_log.csv`
- **Best Models**: `app/resources/models/{subject_id}/best_model.pth`
- **Results DataFrame**: Contains accuracy, training size, and strategy for each subject/fold

## Performance Metrics

- **Primary Metric**: Classification Accuracy
- **Additional Metrics**: Validation loss, training/validation accuracy per epoch
- **Statistical Analysis**: Paired/independent t-tests for strategy comparison

## Key Design Decisions

1. **3-Fold CV**: Chosen over 5-fold to provide larger validation sets (30% vs 20%)
2. **Stratified Splits**: Ensures balanced class distribution in train/val sets
3. **Early Stopping**: Prevents overfitting with patience of 30 epochs
4. **Increased Regularization**: Dropout (0.6) and weight decay (1e-3) to handle small datasets
5. **Combined Sessions**: Subject-dependent training uses all available data per subject

