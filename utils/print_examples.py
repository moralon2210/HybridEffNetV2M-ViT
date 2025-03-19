import torch.nn as nn
import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.getcwd()))

from data_utils.MetGBM_data_handler import MetGBMDataset


def get_channel_idx(dataset, image_name, config):
    """
    Returns the index of the given image name within the dataset's input images list.
    Raises a ValueError if the image name is not found.
    """
    try:
        idx = dataset.params['input_images'].index(image_name)
        return idx
    except ValueError:
        config.logger.error(f"Image '{image_name}' not found in dataset input images list.")
        raise ValueError(f"Image '{image_name}' not found in the list.")


def print_data_example(dataset, df_x, config):
    """
    Displays a random data example from the dataset, including images and metadata.
    """
    config.logger.info("Displaying dataset example...")

    # Randomly select a data point
    item_num = random.randint(0, dataset.__len__() - 1)
    image, target, seg = dataset.__getitem__(item_num)

    patient_number = df_x.iloc[item_num]['Patient number']
    patient_seg = df_x.iloc[item_num]['Seg path']

    config.logger.info(f"Data point #{item_num} from the training set")
    config.logger.info(f"Patient number: {patient_number}")
    config.logger.info(f"Segmentation path: {patient_seg}")
    config.logger.info(f"Class label: {target}")

    # Show image if enabled
    if config.global_params.with_plot:
        for idx, image_name in enumerate(dataset.params['input_images']):
            fig, ax = plt.subplots()
            ax.imshow(255 * (image.permute(1, 2, 0)[:, :, idx]), cmap='gray')
            ax.set_title(f"Sample Image: {image_name} (Patient {patient_number})")
            plt.show()
            plt.close(fig)

    config.logger.info("Dataset example display complete.")
    

