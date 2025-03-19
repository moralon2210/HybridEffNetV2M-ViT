HybridEffNetV2M-ViT

Classification of Glioblastoma and Brain Metastasis Using a Hybrid EfficientNet-Vision Transformer Model

# Introduction

This repository implements a hybrid deep learning model combining EfficientNetV2M and ViT-B16 to classify glioblastoma (GBM) and brain metastases (BM) using MRI images. 
The model integrates EfficientNetV2M and ViT-B16 architectures to leverage the complementary strengths of convolutional neural networks (CNNs) and Vision Transformers (ViTs) 
to improve tumor classification accuracy and generalization across datasets. 
Additionally, the repository supports the standalone use of EfficientNetV2M, ViT-B16, and ResNet-50 as individual architectures.


# Directory Structure

config.py - Imports configuration settings from configs

eval.py - Evaluation utilities

main.py - Main entry point for running the project

inference_predict.py - Inference and prediction script

train.py - Model training script

train_loop.py - Training loop utilities

configs/ - Configuration files

Dataset/ - Dataset files

data_utils/ - Data processing utilities

models/ - Neural network models

requirements/ - Dependency requirements

results/ - Output results and run logs

utils/ - Additional utility functions

weights/ - Pre-trained model weights for all models


# Installation

Clone the repository:

git clone <repository_url>
cd HybridEffNetV2M-ViT


# Install dependencies:

pip install -r requirements.txt


# Dataset

Ensure that the dataset folder and csv file structure matches the expected format as detailed in the dataset directory.


# Configuration

Modify settings in the configs/ directory to adjust training parameters, data paths, and model settings.


# Usage

 1. Train the model:
    python main.py --mode train

 2. Run inference (Evaluation with Ground Truth):
    python main.py --mode inference

 3. Generate predictions (Without Ground Truth):
    python main.py --mode predict


# Results

Model outputs, logs, and performance metrics are automatically saved.
Results are stored in the results/ directory, organized by mode.


# Pre-trained Weights

Available in the weights/ directory

Includes pre-trained weights for EfficientNetV2M, ViT-B16, ResNet-50, and the hybrid model.

These weights are trained over 5 folds and are used during inference and prediction modes to generate results.