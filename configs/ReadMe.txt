This document explains how to configure training, inference, and prediction settings using the JSON configuration files in this directory. 

# User-Editable Configuration Files

There are four primary configuration files:

 1. Train Configuration (train_config.json) - Defines settings for model training.

 2. Inference Configuration (inference_config.json) - Adjusted for evaluation with test data.

 3. Prediction Configuration (predict_config.json) - Used for running inference on unseen data without labels.

 4. Processing Configuration (processing_config.json) - Defines label encoding for categorical targets.	


# Key Configuration Settings

These settings define the essential alterations required to run training, inference, and prediction. 
Users must specify the model type, hardware allocation, and dataset locations on on the relevant mode config:

 1. Model Selection on "model": Choose from "EffNetV2M", "EffNetV2M_VitB16" (hybrid model), "ResNet50", or "VitB16".

 2. Device Assignment on "device:gpu": Set the GPU for training and inference. For example, use "cuda:0" to run on GPU 0.

 3. Dataset Paths on "system_paths": Define paths for dataset folder and csv file for your relevant mode.
    More information on the expected dataset structure can be found on the "Dataset" directory. 


# Optional Configuration Parameters

All settings in the configuration files are required for the model to function correctly. 
However, certain parameters can be adjusted based on user needs to optimize performance or resource allocation.

 1. Training (train_config.json)

   - Model Parameters (model_params): These parameters are required for model functionality but can be left as default if no modifications are needed. 
     They include optimizer, learning_rate, dropout and more.

   - Augmentations (dataset_params): Controls transformations such as vertical_flip, horizontal_flip, rotate_limit, etc.

 2. Inference & Prediction (inference_config.json & predict_config.json)

   - Reduced Parameters: Augmentations and training-specific parameters are removed.

   - Evaluation-Specific Settings: Includes save_preds_per_fold for tracking predictions across cross-validation folds.


# Processing Configuration

The processing_config.json file contains predefined mappings for label encoding. This ensures that categorical targets are correctly interpreted by the model.


# Trained Weights Configuration

The trained_weights_config.json contains predefined settings for the trained models used in inference and prediction modes. 
These values should not be modified, as they represent the optimal configurations derived from training.