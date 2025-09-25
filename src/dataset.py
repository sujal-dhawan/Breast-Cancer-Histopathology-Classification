# src/dataset.py

import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.model_selection import train_test_split

# Use AUTOTUNE for dataset pipelines
AUTOTUNE = tf.data.AUTOTUNE

# --- Paths ---
DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "BreaKHis_v1")

# --- Auto-detect folds file ---
fold_files = glob.glob(os.path.join(DATA_DIR, "Folds.*")) + \
             glob.glob(os.path.join(DATA_DIR, "BreaKHis_v1", "Folds.*"))

if not fold_files:
    raise FileNotFoundError(
        f"❌ No folds file found. Expected Folds.csv or Folds.xlsx inside {DATA_DIR} or {DATA_DIR}/BreaKHis_v1"
    )

fold_file = fold_files[0]
print(f"\n✅ Using folds file: {fold_file}\n")

# --- Load folds file ---
if fold_file.endswith(".csv"):
    data = pd.read_csv(fold_file)
elif fold_file.endswith(".xlsx"):
    data = pd.read_excel(fold_file)
else:
    raise ValueError(f"❌ Unsupported folds file format: {fold_file}")


# ✅ UPDATE class_names to the 8 subtypes from the reference document
class_names = [
    'adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',
    'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma'
]
# Map filename abbreviations to full class names
label_map = {
    'A': 'adenosis', 'F': 'fibroadenoma', 'PT': 'phyllodes_tumor', 'TA': 'tubular_adenoma',
    'DC': 'ductal_carcinoma', 'LC': 'lobular_carcinoma', 'MC': 'mucinous_carcinoma', 'PC': 'papillary_carcinoma'
}

# --- Process Dataset ---
data = data.rename(columns={'filename': 'path'})
data['path'] = data['path'].apply(lambda x: os.path.join(IMG_DIR, *x.split('/')[1:]))
data['filename'] = data.path.apply(lambda x: os.path.basename(x))

# ✅ UPDATE the label extraction logic to handle 8 classes
def get_label_from_filename(filename):
    match = re.search(r'SOB_.\_([A-Z]+)\-.*', filename)
    if match:
        abbreviation = match.group(1)
        return label_map.get(abbreviation, 'unknown')
    return 'unknown'

data['label'] = data.filename.apply(get_label_from_filename)
data = data[data['label'] != 'unknown'] # Remove rows with unknown labels
data['label_int'] = data.label.apply(lambda x: class_names.index(x))


print("\n--- First few rows of new 8-class dataset ---")
print(data.head(3))

# --- Split dataset into train/valid/test ---
# ✅ UPDATE the data split to a 70/15/15 ratio
# 70% for training, 30% for temp
train_df, temp_df = train_test_split(data, test_size=0.30, random_state=42, stratify=data['label'])
# Split the 30% temp into two 15% halves for validation and testing
valid_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])


print("\n--- Data Split Counts ---")
print(f"Training set:   {len(train_df)} samples")
print(f"Validation set: {len(valid_df)} samples")
print(f"Test set:       {len(test_df)} samples")
print("\nTraining set counts (Before Upsampling):")
print(train_df.label.value_counts())


# --- Up-sampling training dataset to balance classes ---
max_count = train_df.label.value_counts().max()
print(f"\nBalancing all training classes to {max_count} samples...")
train_df = train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=max_count, replace=True, random_state=42))

print("\nBalanced training set counts (After Upsampling):")
print(train_df.label.value_counts())