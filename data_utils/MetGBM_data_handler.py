# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:21:01 2024

@author: mora
"""
# Data Augmentation for Image Preprocessing
from albumentations import (ToFloat, Normalize, Resize, VerticalFlip, HorizontalFlip, Compose,
                            RandomBrightnessContrast, HueSaturationValue, GaussNoise,
                            Rotate, RandomResizedCrop, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2
from data_utils.DataPreProcess import image_norm
from torch.utils.data import Dataset
import platform
import nibabel as nib 
import os
import numpy as np
import pandas as pd

class MetGBMDataset(Dataset):
    
    def __init__(self, df_x, config,dataset_type,df_y=None,transforms=None):
        
        """
        Custom dataset for tumor images.
        
        """
        self.df_x,self.df_y = df_x,df_y
        self.config = config
        self.dataset_type = dataset_type
        
        self.dataset_path = config.paths.dataset_paths[dataset_type]['dataset_path']
            

        
        #patient and seg (tumor) list
        self.patients_list = df_x.iloc[:][self.config.csv_columns.patient_col].values.tolist()
        self.segs_list = df_x.iloc[:][self.config.csv_columns.seg_col].values.tolist()
        
        
        self.tumor_count = self.get_tumor_count()
        
        #Normalization
        self.norm = image_norm
        
        # Set transformations
        if transforms is not None:
            self.transform = transforms
        else:
            self.transform = self.default_transforms()
            
    def get_tumor_count(self):
        #returns the total number of unique tum# We should - If train/valid: image + class | If test: only image. added for now as its needed later in our pipeline for evaluators in the dataframe
        df_patient_seg = self.df_x[[self.config.csv_columns.patient_col,self.config.csv_columns.seg_col]]
        tumor_count = len(df_patient_seg.groupby(list(df_patient_seg.columns)))
        
        return tumor_count
        
        
    def default_transforms(self):
        """
        Default transformations for the dataset.
        """
        if self.dataset_type == self.config.train:
            return Compose([
                Resize(self.config.dataset_params.resize_height, self.config.dataset_params.resize_width),
                HorizontalFlip(p=self.config.dataset_params.horizontal_flip),
                VerticalFlip(p=self.config.dataset_params.vertical_flip),
                ToTensorV2()
            ])
        else:
            return Compose([
                Resize(self.config.dataset_params.resize_height, self.config.dataset_params.resize_width),
                ToTensorV2()
            ])
    
    def image_load(self,image_folder):
        """
        Load the image from path 
        
        """
    
        if platform.system() == 'Linux':
            image_folder = image_folder.replace('\\','/')
                                                
        image_full_path = os.path.join(self.dataset_path,image_folder)
        image_nii = nib.load(image_full_path)
        image = image_nii.get_fdata()
             
        
        return image
    
    def get_main_slice_idx(self,seg_mask,idx):
        
        """
       Get the slice index with the most non-zero values.

       """
    

        # importing slice idx from df
        main_slice_idx = int(self.df_x.iloc[idx][self.config.csv_columns.slice_idx_col])
        
        return main_slice_idx
        
        
            
    def get_tumor_voxel_bb(self,seg_mask,main_slice_idx):
        
        """
        Get bounding box around the tumor in the segmentation mask.
    
        """
            
        seg_main_slice = seg_mask[:,:,main_slice_idx]
        rows,cols = np.nonzero(seg_main_slice)
        
        #find height min max
        min_row = np.max([np.min(rows)-self.config.dataset_params.pixel_around_tumor,0])
        max_row = np.min([np.max(rows)+self.config.dataset_params.pixel_around_tumor,np.shape(seg_mask)[0]])
    
        #find width min max
        min_col = np.max([np.min(cols)-self.config.dataset_params.pixel_around_tumor,0])
        max_col = np.min([np.max(cols)+self.config.dataset_params.pixel_around_tumor,np.shape(seg_mask)[1]])
        
        # save bb
        #bb = ([min_row,max_row+1],[min_col,max_col+1],[min_slice,max_slice+1]) #one image scenario
        bb = ([min_row,max_row+1],[min_col,max_col+1])
        
        return bb
    
    
    def __len__(self):
        return len(self.df_x)
    
    def __getitem__(self, idx):
        
        ## load tumor segmentation and get bb
        seg_folder = str(self.df_x.iloc[idx][self.config.csv_columns.seg_col])
        seg_image = self.image_load(seg_folder)
        
        main_slice_idx = self.get_main_slice_idx(seg_image,idx)
        
        #get tumor voxel bounding box from segmentation mask and params bb shape
        bb = self.get_tumor_voxel_bb(seg_image,main_slice_idx)
        
        ## load and handle images
        #process each image seperatly 
        for i in np.arange(self.config.image_in_channels):
            image_name = self.config.input_images[i]               
            image_folder = str(self.df_x.iloc[idx][image_name])
            image = self.image_load(image_folder)
            
            if i==0:
                image_shape = np.shape(image)
                #init images array (images stack)
                image_stack = np.zeros((image_shape[0],image_shape[1],self.config.image_in_channels)) #H x W x channels 
            
            image = self.norm(image)
            image = image[:,:,main_slice_idx] 
            
            # save processed images to images array
            image_stack[:,:,i] = image
        
        # slice image array
        image_patch = image_stack[bb[0][0]:bb[0][1],bb[1][0]:bb[1][1],:]
        image_patch = image_patch.astype(np.float32)
        
        # Apply transforms
        image_patch = self.transform(image=image_patch)
        # Extract image from dictionary
        image_patch = image_patch['image']
            
        
        #for mean\majority voting in 2D
       
        patient = self.df_x.iloc[idx][self.config.csv_columns.patient_col]
        seg = self.df_x.iloc[idx][self.config.csv_columns.seg_col]

        # Handle labels for train, inference and predict
        label = -1 if self.df_y is None else self.df_y.iloc[idx]
    
        return image_patch, label, seg