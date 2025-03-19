#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:45:54 2025

@author: karinmoran
"""

import argparse
import torch
import train
import inference_predict
import json

def parse_args():
    parser = argparse.ArgumentParser(description="""Classification of glioblastoma and brain metastasis using a hybrid
EfficientNet-Vision Transformers model: Train or Run Inference""")
    
    parser.add_argument(
        "--mode", type=str, choices=["train", "inference","predict"], required=True,
        help="Choose 'train' to train the model,'inference' to run inference or 'predict' to get tumor predictions ")
    

    return parser.parse_args()

def main():
    args = parse_args()  

    if args.mode == "train":
        
        print(f"Starting training")
        train.main(args.mode)
    
    elif args.mode in ["inference","predict"]:
        print(f"Running {args.mode}")
        inference_predict.main(args.mode)



if __name__ == "__main__":
    main()