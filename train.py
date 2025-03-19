# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:27:59 2024

@author: mora
"""
# %[code] {"_kg_hide-input":true,"_kg_hide-output":true,"jupyter":{"outputs_hidden":false}}

# config
from config import Config

# utils
from data_utils.DataPreProcess import df_processing
from data_utils.SplitTrainValTest import split_train_val
from data_utils.MetGBM_data_handler import MetGBMDataset
from utils.print_examples import print_data_example
from utils.results_util import get_results_folder,save_metrices_results
from eval import evaluate_oof_main
from time import gmtime, strftime
from utils.system_utils import set_seed
import train_loop
from utils.logger import get_logger

# Basics
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


    
def main(mode):
    """Main function for training and evaluation."""

    # Upload configs
    config = Config(mode)

    # Create results and output directory for current run
    config.add_curr_run_folder(get_results_folder(config.paths.results_path))
    
    # Init logger
    config.logger = get_logger(mode, config.curr_run_results_path)

    config.logger.info(f"Run number: {config.curr_run_results_folder}")
    config.run_start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # Dataset processing
    config.logger.info("Loading train dataset...")
    df_train_pre = pd.read_csv(config.paths.dataset_paths[config.train]['csv_path'])
    df_train, df_x_train, df_y_train = df_processing(df_train_pre, config.train, config, mode)
    config.train_len = len(df_x_train)

    # Create dataset class and log sample example
    dataset = MetGBMDataset(df_x_train, config, config.train, df_y_train)
    #print_data_example(dataset, df_x_train, config)
    
    config.logger.info(f"Splitting data into {config.model_params.k} folds...")
    # Set seed for fold reproducibility
    set_seed()

    # Divide into folds
    folds = split_train_val(config.model_params.k, df_x_train, df_y_train)


    # Train model
    folds_oof_mean_preds_dict, folds_oof_tumor_label_dict, folds_best_model = train_loop.main(
        folds, df_x_train, df_y_train, config, mode)

    # Evaluation of OOF and test predictions
    print('\n')
    config.logger.info("Evaluating validation results...")
    oof_metrices_dict, config.optimal_threshold = evaluate_oof_main(folds_oof_mean_preds_dict, 
                                                                    folds_oof_tumor_label_dict, config)

    config.logger.info("Saving validation results...")
    oof_results = {
        'run number': config.curr_run_results_folder,
        'model type': config.model_type_str,
        'optimal oof threshold': config.optimal_threshold,
        **config.model_params.dict,
        **oof_metrices_dict}
    
    save_metrices_results(oof_results,config.oof, config)

    
    
if __name__ == "__main__":
    mode = 'train'
    main(mode)
    
    
                
    
                
            
    

    
                       
                       
                        
            
                        
