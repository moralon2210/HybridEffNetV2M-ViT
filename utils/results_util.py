#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:55:21 2024

@author: karinmoran
"""
import os 
import numpy as np
import pandas as pd
import json

def get_results_folder(results_base_path):
    """Set up and return the results directory for the current run."""
    # Create results directory if it doesn't exist
    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)
        curr_run_results_folder = '1'  # Start from folder 1 if none exist

    # If directory exists but is empty, start from folder 1
    elif not os.listdir(results_base_path):
        curr_run_results_folder = '1'

    # Otherwise, increment from the last numbered folder
    else:
        folders = os.listdir(results_base_path)
        folders_num = np.sort(np.array([int(folder) for folder in folders if os.path.isdir(os.path.join(results_base_path, folder))]))
        curr_run_results_folder = str(folders_num[-1] + 1)

    return curr_run_results_folder


def create_results_df(csv_results_path,config):
    """Create and return a DataFrame from the results file, or initialize an empty DataFrame if the file doesn't exist."""
    # Load existing results if the file exists
    if os.path.exists(csv_results_path):
        results_df = pd.read_csv(csv_results_path)
    else:
        # Initialize an empty DataFrame if no file is found
        results_df = pd.DataFrame()

    return results_df


def save_predictions(mode, segs_list, test_mean_preds,test_mean_preds_ths,test_tumor_labels,
                     folds_test_preds_mean_dict, config, dataset_type):
    """Process and save predictions based on the inference mode."""
    # Generate string predcitions
    rever_label_encoding = {v: k for k, v in config.label_encoding.items()}
    test_pred_str = [rever_label_encoding[i] for i in test_mean_preds_ths]

    # Prepare prediction results dictionary
    results = {config.csv_columns.seg_col: segs_list, "Prob pred":np.round(test_mean_preds,6),
               "Final pred num": test_mean_preds_ths, "Final pred str":test_pred_str}
    
    if mode=='inference':
        test_label_str = [rever_label_encoding[i] for i in test_tumor_labels]
        results.update({'Label str':test_label_str})

    # Include predictions from each fold if configured
    if config.global_params.save_preds_per_fold:
        results.update(folds_test_preds_mean_dict)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).sort_values(by=config.csv_columns.seg_col)

    # Ensure the directory for saving predictions exists
    save_path = config.preds_results_path

    # Save predictions to CSV file
    results_df.to_csv(save_path, index=False)
    config.logger.info(f"{dataset_type} predictions were saved to: {save_path}")

    return results_df


def save_metrices_results(results, dataset_type, config):
    """Save evaluation metrics and hyperparameters to the results file."""
    # Define the path to save metrics
    results_path = config.paths.dataset_paths[dataset_type]['results_metrices_path']

    # Load existing results or create a new DataFrame
    results_df = create_results_df(results_path,config)
    results_df_new = pd.DataFrame([results])

    # Append the new results to the existing DataFrame
    results_df = pd.concat([results_df, results_df_new], ignore_index=True)

    # Save the updated DataFrame to CSV
    results_df.to_csv(results_path, index=False)

    config.logger.info(f"Metrics and hyperparameters saved in {results_path}")
    
    

    
    
    