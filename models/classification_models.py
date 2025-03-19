#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:42:20 2024

@author: karinmoran
"""
from models.embeddings import ResNet50ImageEmbedd, EfficientImageEmbedd,VitB16ImageEmbedd,EfficientV2MImageEmbedd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def embedd_dict():
    """
    Creates a dictionary mapping vision model names to their corresponding embedding classes.
    """
    embedding_models = {
        "EffNetV2M": EfficientV2MImageEmbedd,
        "ResNet50": ResNet50ImageEmbedd,
        "VitB16": VitB16ImageEmbedd
    }
    
    return embedding_models
    
    
def pick_embedd(vision_model, image_in_channels, device,transfer_learning=False):
    """
    Selects the appropriate embedding model based on the vision model name.
    If the model name is in the predefined dictionary, it is directly used.
    Otherwise, if the model starts with 'EffNet_', it extracts the specific variant.
    Raises an error if the model name is not recognized.
    """
    embedding_models = embedd_dict()
    
    if vision_model in embedding_models:
        return embedding_models[vision_model](image_in_channels, transfer_learning)
    
    elif vision_model.split('_')[0] == 'EffNet':
        b_type = vision_model.split('_')[1].lower()
        return EfficientImageEmbedd(image_in_channels,
                                    b_type, transfer_learning=transfer_learning).to(device)
    
    else:
        raise ValueError(f"Unknown vision model: {vision_model}")

        
class HybridVisionClassification(nn.Module):
    """
    A hybrid deep learning model that supports one or multiple vision models for classification.
    Extracted embeddings from different vision models are concatenated before passing through a classification head.
    """
    
    def __init__(self, config, output_size):
        
        
        super().__init__()
        self.output_size = output_size
        self.in_channels = config.image_in_channels
        self.with_transfer_learning = getattr(config.model_params, "with_transfer_learning", False)
        self.dropout = getattr(config.model_params, "dropout", 0)
        
        # Create a list of embedding models based on the provided configuration
        self.image_embedds = nn.ModuleList([
            pick_embedd(model_name, config.image_in_channels, config.device,self.with_transfer_learning)
            for model_name in config.model_type
        ])
        
        # Calculate the total embedding size by summing feature sizes from all models
        self.embedds_joint_len = sum([image_embedd.embedd_len for image_embedd in self.image_embedds])

        # Define the classification head with dropout before the last fully connected layer
        self.classification = nn.Sequential(
            nn.Linear(self.embedds_joint_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, output_size)
        )
        
    def forward(self, image, csv_data=None, prints=False):
        
        if prints:
            print(f"Input Image shape: {image.shape}")

        # Extract features from all vision models and concatenate them along the feature dimension
        image_features = [model(image) for model in self.image_embedds]
        image_features = torch.cat(image_features, dim=1)

        if prints:
            print(f"Concatenated Image Features shape: {image_features.shape}")

        # Pass the concatenated features through the classification head
        out = self.classification(image_features)

        if prints:
            print(f"Output shape: {out.shape}")

        return out