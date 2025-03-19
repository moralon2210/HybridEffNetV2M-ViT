#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:36:51 2024

@author: karinmoran
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc, recall_score,
    precision_score, confusion_matrix, f1_score, brier_score_loss
)
import os

def calculate_metrics(preds, tumor_labels, dataset_type):
    """
    Calculates classification metrics.
    """
    acc = accuracy_score(tumor_labels, preds)
    f1 = f1_score(tumor_labels, preds)
    precision = precision_score(tumor_labels, preds)
    recall = recall_score(tumor_labels, preds)

    metrics_dict = {
        f"acc": acc,
        f"f1 score": f1, 
        f"precision": precision, 
        f"recall": recall
    }
    
    return metrics_dict

def plot_save_roc_curve(y_train, oof_preds, config):
    """
    Plots and saves the ROC curve.
    """
    auc_mean = roc_auc_score(y_train, oof_preds)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_train, oof_preds)

    # Find optimal cutoff using Youden’s J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    config.logger.info(f"\nOptimal threshold is {optimal_threshold}. Will be used for test predictions!")
    
    # Plot ROC curve
    roc_auc = auc(fpr, tpr)
    matplotlib.use('Agg')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc='lower right')
    
    # Save the plot
    plt.savefig(os.path.join(config.curr_run_results_path, "ROC.png"))
    if config.global_params.with_plot:
        plt.show()

    return auc_mean, optimal_threshold

def plot_save_confusion_mat(labels, preds, dataset_type, config):
    """
    Plots and saves the confusion matrix.
    """
    # Reverse label encoding to get class names
    label_encoding_flipped = {value: key for key, value in config.label_encoding.items()}
    
    # Compute confusion matrix
    cf_matrix = confusion_matrix(labels, preds)

    # Get class names
    neg_class = label_encoding_flipped[0]
    pos_class = label_encoding_flipped[1]
    
    # Extract values from confusion matrix
    tn, fp, fn, tp = cf_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    misclass = fp + fn  # Total misclassified samples

    # Format confusion matrix labels
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{:,}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.1%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    # Plot confusion matrix
    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.clf()
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Oranges',
                xticklabels=[neg_class, pos_class], 
                yticklabels=[neg_class, pos_class], cbar=False)
    
    plt.rcParams.update({'font.size': 15})
    ax.tick_params(axis='both', labelsize=15)
    ax.set_title(f"Confusion Matrix: {dataset_type}", fontsize=20)
    
    # Display accuracy and misclassification rate
    metrics_text = f'Accuracy= {accuracy:.2f}, Misclass: {misclass:.2f}'
    plt.text(0.5, -0.1, metrics_text, ha='center', va='center', fontsize=15, transform=ax.transAxes)
    
    # Save confusion matrix plot
    cm_save_path = os.path.join(config.curr_run_results_path, f"Confusion matrix for {dataset_type}.png")  
    plt.savefig(cm_save_path)
    config.logger.info(f"\nConfusion matrix for {dataset_type} was saved to: {cm_save_path}")
    
    if config.global_params.with_plot:
        plt.show()
    
    plt.close(fig)

def evaluate_oof_main(folds_oof_mean_preds_dict, folds_oof_tumor_label_dict, config):
    """
    Evaluates Out-of-Fold (OOF) predictions.
    """
    dataset_type = config.oof
    oof_metrics_dict = {}

    # Concatenate predictions and labels
    oof_mean_preds = np.concatenate(list(folds_oof_mean_preds_dict.values()))
    oof_ths_mean_preds = (oof_mean_preds > config.oof_threshold).astype(int)
    oof_labels = np.concatenate(list(folds_oof_tumor_label_dict.values()))

    # Compute ROC curve and find optimal threshold
    auc_mean, optimal_threshold = plot_save_roc_curve(oof_labels, oof_mean_preds, config)
    oof_metrics_dict['auc'] = auc_mean 

    # Compute Brier score
    brier_score = brier_score_loss(oof_labels, oof_mean_preds)
    oof_metrics_dict['brier score'] = brier_score

    # Compute classification metrics
    metrics_dict = calculate_metrics(oof_ths_mean_preds, oof_labels, dataset_type)
    oof_metrics_dict.update(metrics_dict)

    # Plot and save confusion matrix
    plot_save_confusion_mat(oof_labels, oof_ths_mean_preds, dataset_type, config)

    return oof_metrics_dict, optimal_threshold

def evaluate_test_main(test_mean_preds, test_mean_preds_ths, test_labels, dataset_type, config):
    """
    Evaluates test predictions using the optimal threshold.
    """
    test_metrics_dict = {}

    # Compute AUC score
    auc_score = roc_auc_score(test_labels, test_mean_preds)
    test_metrics_dict['auc'] = auc_score

    # Compute Brier score
    brier_score = brier_score_loss(test_labels, test_mean_preds)
    test_metrics_dict['brier score'] = brier_score

    # Compute classification metrics
    metrics_dict = calculate_metrics(test_mean_preds_ths, test_labels, dataset_type)
    test_metrics_dict.update(metrics_dict)

    # Plot and save confusion matrix
    plot_save_confusion_mat(test_labels, test_mean_preds_ths, dataset_type, config)

    return test_metrics_dict
    
