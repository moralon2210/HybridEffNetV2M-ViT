# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:36:18 2024

@author: mora
"""
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, train_test_split
import numpy as np

def split_train_test(df, patient_col, seg_col, target_col, test_size=0.1, random_state=42):
    """
    Splits the dataset into training and test sets, ensuring that data points from the same patient 
    are grouped together in the same set and that the distribution of tumor counts is similar in both sets.

    Args:
    df (pd.DataFrame): The input dataframe containing the data.
    patient_col (str): The column name representing the patient identifier.
    target_col (str): The column name representing the target variable.
    test_size (float): The proportion of the dataset to include in the test split (default is 0.05).
    random_state (int): The seed used by the random number generator (default is 42).

    Returns:
    tuple: Four dataframes (df_x_train, df_y_train, df_x_test, df_y_test) representing the features and target 
           variables for the training and test sets.
    """
    print('\n### Dividing dataset to train and test')
    
    tumor_counts =   df.groupby(patient_col)[seg_col].nunique()      
    
    # Split patient IDs into train and test sets
    # Ensure similar distribution of tumor counts while maintaining each patient in the same data subset
    # the first is achieved using stratify while the latter by using tumor_counts.index (which is patient number)
    
    train_ids, test_ids = train_test_split(tumor_counts.index, test_size=test_size, 
                                           stratify=tumor_counts, random_state=random_state)
    
    # Assign images to sets based on patient IDs
    df_train = df[df[patient_col].isin(train_ids)].reset_index(drop=True)
    df_test = df[df[patient_col].isin(test_ids)].reset_index(drop=True)
    
    #train
    df_x_train = df_train.drop(columns = target_col)
    df_y_train = df_train[target_col]
    
    #test
    df_x_test = df_test.drop(columns = target_col)
    df_y_test = df_test[target_col]
    
    print('Len of train is: {}'.format(len(df_x_train)))
    print('Len of test is: {}'.format(len(df_x_test)))


    return df_x_train, df_y_train, df_x_test, df_y_test

def split_train_val(k,df_x_train,df_y_train):
    """
    Splits the training data into stratified folds for cross-validation.

    This function uses StratifiedGroupKFold to create k folds for cross-validation, ensuring that 
    the percentage of samples for each class is preserved as much as possible in each fold while maintaining 
    non-overlapping groups between splits. The 'Patient number' column is used for grouping to ensure 
    each patient appears in only one subset. Crucial to prevent data leakge between folds. 
    
    Args:
        k (int): The number of folds.
        df_x_train (DataFrame): The training data features.
        df_y_train (DataFrame): The training data labels.
        
    Returns:
        generator: A generator yielding train/validation indices for each fold.
    """
   
    group_fold = StratifiedGroupKFold(n_splits = k,shuffle = False)
    
    train_len = len(df_x_train)
    
    # Generate indices to split data into training and validation set.
    #Group folds provide int positions, not index
    folds = group_fold.split(X = np.zeros(train_len), 
                             y = df_y_train.values, 
                             groups = df_x_train['Patient number'])
    
    return folds