
import pandas as pd
import numpy as np
import torch

def get_model_preds(model, data_loader, len_preds, device,has_labels):
    """
    Gets model predictions and handles cases where labels may be missing.

    """
    model.eval()
    
    preds_list = []
    
    labels_list = [] if has_labels else None  # Only create labels list if labels exist

    with torch.no_grad():
        for images, labels, segs in data_loader:
            images = images.to(device).float()
            out = model(images)
            preds = torch.sigmoid(out)

            # Store predictions
            preds_list.extend(preds.flatten().cpu().numpy().astype(float))

            # Store labels only if available
            if has_labels:
                labels_list.extend(labels.cpu().tolist())

    return preds_list, labels_list

def agg_predictions_per_tumor(segs_list, predictions, labels=None):
    """
    Aggregates predictions per tumor by computing the mean probability for each unique segmentation.

    """
    seg_column = "Seg path"
    preds_column = "predictions"
    label_column = "labels"

    # Create DataFrame for grouping
    df_combined = pd.DataFrame({seg_column: segs_list, preds_column: predictions})

    # Only add labels if they exist
    if labels is not None:
        df_combined[label_column] = labels

    # Compute mean of prediction probabilities per segmentation
    df_agg_score = df_combined.groupby(seg_column, as_index=False).mean()
    agg_mean_predictions = df_agg_score[preds_column].values

    # Handle tumor labels correctly (only if available)
    tumor_labels = df_agg_score[label_column].values.astype(int) if labels is not None else None

    # Return aggregated predictions, labels (if available), and segmentation paths
    return agg_mean_predictions, tumor_labels, df_agg_score[seg_column].tolist()





