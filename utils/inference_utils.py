import os
import shutil
import gdown
import pandas as pd
import json
from data_utils.DataPreProcess import df_processing
import numpy as np
import glob
import platform
import ast
import torch
from eval import evaluate_test_main
from data_utils.MetGBM_data_handler import MetGBMDataset
from torch.utils.data import DataLoader
from utils.model_utils import agg_predictions_per_tumor, get_model_preds
from utils.results_util import create_results_df, save_metrices_results, save_predictions

#pd.set_option('future.no_silent_downcasting', True)

def download_weights(config):
    model_weights_id_dict = {'VitB16':'1U14ryUmjHyeE9B1DjPC9Pnn-e0-Evp3P',
                           'ResNet50':'1U4drJRssBFysdq6HzAFld_CBqp4j0YmX',
                           'EffNetV2M_VitB16':'1U5ZwRyhAQeKo0wAnNIC6VpZSXzP5oNhI',
                           'EffNetV2M':'1U7jO-Lntqf9fa26Xb834z0d2yjaZ-NyF'}
    
    # Check if exists
    if os.path.exists(config.weights_folder):
        if len(glob.glob1(config.weights_folder,'*.pth'))==5:
            config.logger.info(f"{config.model_type_str} weights folder was found..")
            return
        else:
            # Current folder content is only partial, delete and re-download
            shutil.rmtree(config.weights_folder)
                                                     
    # Download
    model_id = model_weights_id_dict[config.model_type_str]
    path = f"https://drive.google.com/drive/folders/{model_id}"
    config.logger.info(f"Downloading {config.model_type_str} weights from web..")
    gdown.download_folder(path, output=config.weights_folder, quiet=False, use_cookies=False)
    

def get_fold_test_preds(model, test, test_loader, test_len, test_preds_len, config, has_labels):
    """Get predictions per slice and aggregate predictions per tumor."""
    test_preds, test_labels = get_model_preds(model, test_loader, test_len, config.device, has_labels)
    agg_mean_predictions, test_tumor_labels, segs = agg_predictions_per_tumor(test.segs_list, test_preds, test_labels)
    return agg_mean_predictions, test_tumor_labels, segs

def get_fold_test_preds(model, test, test_loader, test_len, test_preds_len, config, has_labels):
    """Get predictions per slice and aggregate predictions per tumor."""
    test_preds, test_labels = get_model_preds(model, test_loader, test_len, config.device, has_labels)
    agg_mean_predictions, test_tumor_labels, segs = agg_predictions_per_tumor(test.segs_list, test_preds, test_labels)
    return agg_mean_predictions, test_tumor_labels, segs

def run_inference(df_x_test, fold_models, dataset_type, config, mode, df_y_test=None):
    """Run inference using each fold model and collect predictions."""
    has_labels = df_y_test is not None
    folds_test_preds_mean_dict = {}  # Store predictions for each fold

    # Prepare dataset and data loader
    test_dataset = MetGBMDataset(df_x_test, config, dataset_type, df_y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.model_params.test_batch_size, shuffle=False,
                             num_workers=config.model_params.num_workers)

    test_len = len(df_x_test)
    test_preds_len = test_dataset.tumor_count

    # Loop through each fold model
    print('\n')
    for i, fold_model in enumerate(fold_models):
        config.logger.info(f"Getting predictions for fold {i + 1}")
        fold_model = fold_model.to(config.device)
        fold_model.eval()

        # Get predictions for current fold
        fold_test_preds_mean, test_tumor_labels, segs_list = get_fold_test_preds(fold_model, test_dataset, test_loader,
                                                                                 test_len, test_preds_len, config, has_labels)

        # Debug: Ensure labels and segment IDs match across folds
        if test_tumor_labels is not None:
            if i == 0:
                prev_test_tumor_labels, prev_segs_list = test_tumor_labels, segs_list
            assert np.array_equal(test_tumor_labels, prev_test_tumor_labels), "Mismatch in labels between folds!"
            assert np.array_equal(segs_list, prev_segs_list), "Mismatch in segmentation IDs between folds!"
            prev_test_tumor_labels, prev_segs_list = test_tumor_labels, segs_list

        # Store predictions for current fold
        folds_test_preds_mean_dict[f"Fold {i + 1}"] = list(fold_test_preds_mean.astype(float))

    return folds_test_preds_mean_dict, test_tumor_labels, segs_list

def get_final_test_predictions(folds_test_preds_mean_dict, config):
    """Compute the final test predictions using the mean predictions from all folds."""
    test_mean_preds = np.mean(np.array(list(folds_test_preds_mean_dict.values())), axis=0)
    test_mean_preds_ths = (test_mean_preds > config.optimal_threshold).astype(int)  # Apply threshold
    return test_mean_preds, test_mean_preds_ths

def main(dataset_type, fold_models, config, mode):
    """Main function to run inference on the test dataset."""
    print('\n')
    config.logger.info("Loading test dataset...")
    df_test_pre = pd.read_csv(config.paths.dataset_paths[dataset_type]['csv_path'])
    df_test, df_x_test, df_y_test = df_processing(df_test_pre, dataset_type, config, mode)

    # Run inference using fold models
    folds_test_preds_mean_dict, test_tumor_labels, segs_list = run_inference(df_x_test, fold_models, dataset_type,
                                                                             config, mode, df_y_test)

    # Compute final test predictions
    config.logger.info('Getting mean voting..')
    test_mean_preds, test_mean_preds_ths = get_final_test_predictions(folds_test_preds_mean_dict, config)

    # Save predictions
    save_predictions(mode, segs_list, test_mean_preds,test_mean_preds_ths, test_tumor_labels,
                     folds_test_preds_mean_dict, config, dataset_type)

    # Evaluate predictions if labels are available
    if mode == 'inference':
        print('\n')
        config.logger.info("Evaluating test results...")
        test_metrices_dict = evaluate_test_main(test_mean_preds, test_mean_preds_ths, test_tumor_labels, dataset_type, config)

        # Compile results and save
        test_results = {'run number': config.curr_run_results_folder, 'model type': config.model_type_str,
                        'optimal oof threshold': config.optimal_threshold , **test_metrices_dict,
                        **config.model_params.dict}

        save_metrices_results(test_results, dataset_type, config)

        config.logger.info(f"Inference completed: {dataset_type}")
        
        


        


                   
    

        

