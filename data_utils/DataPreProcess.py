# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:53:34 2024

@author: mora
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from intensity_normalization.normalize.zscore import ZScoreNormalize
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
import matplotlib.pyplot as plt
import sys


def divide_df_x_y(df, target_col, mode, config):
    """
    Splits the dataset into input features and target labels, handling missing target_col cases.
    Exits the script if target_col is missing in train or inference mode.
    """
    if mode != 'predict':
        df_x = df.drop(columns=[target_col])  # Drop target column
        df_y = df[target_col].astype(int)     # Convert labels to int
    else:  # For prediction mode (df_y is always None)
        df_x = df
        df_y = None

    return df_x, df_y


def df_processing(df, dataset_type, config, mode):
    """
    Cleans and preprocesses the dataset, handling missing values,
    removing unnecessary columns, encoding target labels,
    and analyzing data distribution.
    """
    config.logger.info(f"Processing dataset: {dataset_type}")
    initial_count = df.shape[0]

    # Remove rows with missing input images
    df.replace('NA', pd.NA, inplace=True)
    df.dropna(subset=config.input_images, inplace=True)
    final_count = df.shape[0]
    removed_count = initial_count - final_count
    if removed_count > 0:
        config.logger.info(f"Removed {removed_count} rows due to missing input images.")

    # Analyze target label distribution
    target_counts = df[config.csv_columns.target_col].value_counts()
    for target, count in target_counts.items():
        config.logger.info(f"Class '{target}': {count} data points")
    config.logger.info(f"Total data points: {len(df)}")

    # Remove rows without target labels for 'train' and 'inference'
    if mode != 'predict':
        if config.csv_columns.target_col not in df.columns:
            config.logger.error(f"Target column '{config.csv_columns.target_col}' not found in the dataset for mode: {mode}. Exiting...")
            sys.exit(1)  # Exit the script with an error status if there are no labels
        else:
            patients_without_target = df[df[config.csv_columns.target_col].isna()]
            if len(patients_without_target) > 0:
                config.logger.warning(f"Removed {len(patients_without_target)} rows without target labels.")
                df.dropna(subset=[config.csv_columns.target_col], inplace=True)

    # Encode target labels
    df[config.csv_columns.target_col] = df[config.csv_columns.target_col].replace(config.label_encoding)
    config.logger.info(f"Assigned encoding {config.label_encoding}")
    config.logger.info(f"Dataset size after processing: {len(df)}")

    # Plot target distribution if enabled
    if config.global_params.with_plot:
        fig, ax = plt.subplots()
        df[config.csv_columns.target_col].value_counts().plot(kind="pie", ax=ax, autopct='%1.1f%%')
        ax.set_title(f"Distribution of {config.csv_columns.target_col}")
        plt.show()
        plt.close(fig)

    config.logger.info(f"Dataset processing completed: {dataset_type}")
    #df = df.iloc[200:300]
    df_x, df_y = divide_df_x_y(df, config.csv_columns.target_col, mode, config)

    return df, df_x, df_y


def image_norm(image):
    """
    Applies Z-score normalization to an image.
    """
    zscore_normalizer = ZScoreNormalize()
    return zscore_normalizer.normalize_image(image)