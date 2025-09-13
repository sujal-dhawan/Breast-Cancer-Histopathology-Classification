# src/dataset.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Use AUTOTUNE for dataset pipelines
AUTOTUNE = tf.data.AUTOTUNE

# --- Paths ---
DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "BreaKHis_v1")  # Remove extra folder duplication

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

class_names = ['benign', 'malignant']

# --- Process Dataset ---
data = data.rename(columns={'filename': 'path'})

# Update to use proper relative paths from IMG_DIR
data['path'] = data['path'].apply(lambda x: os.path.join(IMG_DIR, *x.split('/')[1:]))

data['label'] = data.path.apply(lambda x: 'benign' if 'benign' in x.lower() else 'malignant')
data['label_int'] = data.label.apply(lambda x: class_names.index(x))
data['filename'] = data.path.apply(lambda x: os.path.basename(x))


print("\n--- First few rows of dataset ---")
print(data.head(3))

# --- Visualize dataset distribution ---
sns.displot(data=data, x='label')
print('\nDataset Counts:')
print('Benign    : ', data[data.label == 'benign'].label.count())
print('Malignant : ', data[data.label == 'malignant'].label.count())

# --- Split dataset into train/valid/test ---
# take 300 samples per class for testing
test_df = data.groupby('label').sample(n=300, random_state=42)
train_df = data.drop(test_df.index).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# split remaining into training + validation (80/20)
valid_df = train_df.sample(frac=0.2, random_state=42)
train_df = train_df.drop(valid_df.index).reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

# mark set type
test_df['set'] = 'test'
train_df['set'] = 'train'
valid_df['set'] = 'valid'

# combine for visualization
data_new = pd.concat([train_df, valid_df, test_df])
sns.displot(data=data_new, x='label', col='set')

print("\nTraining set counts:")
print(train_df.label.value_counts())
print("\nValidation set counts:")
print(valid_df.label.value_counts())
print("\nTest set counts:")
print(test_df.label.value_counts())

# --- Up-sampling training dataset to balance classes ---
max_count = np.max(train_df.label.value_counts())

# resample minority class to match majority
train_df = train_df.groupby('label').sample(n=max_count, replace=True, random_state=42)
train_df = train_df.reset_index(drop=True)

print("\nBalanced training set counts:")
print(train_df.label.value_counts())

sns.displot(data=train_df, x='label')

# --- Exports ---
img_dir = IMG_DIR
