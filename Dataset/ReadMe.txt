# This dataset is organized into two main folders:

train/

test/

Each folder contains:

1. A CSV file (train.csv\test.csv) that holds metadata for patient data.

2. A PatientImg/ directory that stores patient-specific MRI scans and segmentation masks.

Example folders and CSV files have been added for clarity.

/dataset_root/
│-- train/
│   │-- train.csv
│   │-- PatientImg/
│       │-- Patient_001/
│           │-- Seg.nii.gz
│           │-- t1ce.nii.gz
│           │-- t2.nii.gz
│-- test/
│   │-- test.csv
│   │-- PatientImg/
│       │-- Patient_002/
│           │-- Seg.nii.gz
│           │-- t1ce.nii.gz
│           │-- t2.nii.gz



# CSV File Structure

Each CSV file (train.csv, test.csv) contains metadata corresponding to patients in the respective folder. Each row represents:

A patient

The tumor segmentation path

MRI image paths

Slice index



# Input Image Configuration

MRI images were axially acquired.

The dataset was trained using t1ce and t2 MRI images, but our project supports other MRI modalities as well.

The specific input images used for training are configurable in the training configuration file.
