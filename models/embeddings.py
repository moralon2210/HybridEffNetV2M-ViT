# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:26:28 2024

@author: mora
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet34, resnet50, vit_b_16, vgg16, efficientnet_v2_m, vit_b_32


class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
        
    def forward(self,x):
        return x
    

class VitB16ImageEmbedd(nn.Module):
    def __init__(self, num_input_channels,transfer_learning=True):
        super().__init__()
        self.transfer_learning = transfer_learning
        self.in_channels = num_input_channels
        
        if self.in_channels!= 3:
            # create mapping layer to be applied prior to conv1
            self.mapping_conv= nn.Conv2d(in_channels=self.in_channels, out_channels=3, 
                                     kernel_size=1, stride=1, padding=0)

        # init weights
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        
        # Load vit model
        self.features = vit_b_16(weights=pretrained_vit_weights) if self.transfer_learning else vit_b_16()
        
        # saving emedding length
        self.embedd_len = self.features.heads.head.in_features
    
        # change the last fc layer ti identity to just get the embedding without prediction
        self.features.heads.head = Identity()
        
    def forward(self, image, prints=False):
        
        if self.in_channels!= 3:
            image = self.mapping_conv(image)
            if prints: print('Post channels mapping image shape:', image.shape)
        
        if prints: print('Input Image shape:', image.shape)
        
        # Extract embedding
        embedd = self.features(image)
        
        return embedd
        
    

class EfficientV2MImageEmbedd(nn.Module):
    def __init__(self, num_input_channels,transfer_learning=True):
        super().__init__()
        self.transfer_learning = transfer_learning
        self.in_channels = num_input_channels
        
        if self.in_channels!= 3:
            # create mapping layer to be applied prior to conv1
            self.mapping_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=3, 
                                     kernel_size=1, stride=1, padding=0)


        # init weights
        pretrained_effV2M_weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
        
        # LOad effV2M model
        self.features = efficientnet_v2_m(weights=pretrained_effV2M_weights) if self.transfer_learning else efficientnet_v2_m()
        
        # saving emedding length
        self.embedd_len = self.features.classifier[1].in_features
    
        # change the last fc layer ti identity to just get the embedding without prediction
        self.features.classifier = Identity()
        
        
    def forward(self, image, prints=False):
        
        if self.in_channels!= 3:
            image = self.mapping_conv(image)
            if prints: print('Post channels mapping image shape:', image.shape)
        
        if prints: print('Input Image shape:', image.shape)
        
        # Extract embedding
        embedd = self.features(image)
        
        return embedd



class ResNet50ImageEmbedd(nn.Module):
    def __init__(self, num_input_channels,transfer_learning=True):
        super().__init__()
        self.in_channels = num_input_channels
        
        
        if self.in_channels!= 3:
            # create mapping layer to be applied prior to conv1
            self.mapping_conv  = nn.Conv2d(in_channels=self.in_channels, out_channels=3, 
                                     kernel_size=1, stride=1, padding=0)
            
        # res50 features (fc to 1000 class output is included)
        self.features = resnet50(pretrained=transfer_learning) # 1000 neurons out
        # saving emedding length
        self.embedd_len = self.features.fc.in_features
        # removing final layer (classificaiton)
        self.features.fc = Identity()
        
        
        
    def forward(self, image, prints=False):
        
        if prints: print('Input Image shape:', image.shape)
        
        #mapping conv
        if self.in_channels  != 3:
            image = self.mapping_conv(image)
        
        # Extract embedding
        embedd = self.features(image)
        if prints: print('Features Image shape:', image.shape)
    
        
        return embedd

# EfficientNet 

class EfficientImageEmbedd(nn.Module):
    def __init__(self,num_input_channels, b_type,transfer_learning=True):
        super().__init__()
        self.efficientnet_b_type = 'efficientnet-' + b_type
        self.in_channels = num_input_channels
        
        if self.in_channels!= 3:
            # create mapping layer to be applied prior to conv1
            mapping_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=3, 
                                     kernel_size=1, stride=1, padding=0)
            
            #batch norm - identical params to the ones on effnet
            bn0 = nn.BatchNorm2d(3, eps=0.001, momentum=0.01, affine=True, track_running_stats=True) 
            
            self.mapping_conv = nn.Sequential(mapping_conv,bn0)
        
        # Define Feature part (IMAGE)
        self.features = EfficientNet.from_pretrained(self.efficientnet_b_type) if transfer_learning else EfficientNet.from_name(self.efficientnet_b_type)
        
        
        # saving emedding length
        self.embedd_len = self.features._fc.in_features
        
        
    def forward(self, image, prints=False):    
        if prints: print('Input Image shape:', image.shape)
        
        #mapping conv
        if self.in_channels  != 3:
            image = self.mapping_conv(image)
            if prints: print('Post channels mapping image shape:', image.shape)
        
        # Extract embedding
        image = self.features.extract_features(image)
        if prints: print('Features Image shape:', image.shape)
        
        #get embedding from feature maps of size (batch_size, num_channels, height, width)
        embedd = F.avg_pool2d(image, image.size()[2:]).reshape(-1, self.embedd_len)
    
        return embedd
    

    

