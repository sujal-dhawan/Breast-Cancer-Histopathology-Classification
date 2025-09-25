# src/debug_pipeline.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import the exact same components as your train.py script
from dataset import train_df, AUTOTUNE, class_names
from helpers import parse_image, augmentor, preprocess_data, resize_image

# --- ✅ CHOOSE WHICH MODEL'S PIPELINE TO DEBUG ---
MODEL_NAME = "ResNet50"  # Options: "ResNet50", "EfficientNetB0"

# Re-create the training pipeline exactly as it is in train.py
print(f"--- Building the data pipeline for {MODEL_NAME}... ---")
train_loader = tf.data.Dataset.from_tensor_slices((train_df.path, train_df.label_int))
image_size = 224
batch_size = 16 # Use a smaller batch for easier inspection

train_ds = (
    train_loader.shuffle(1000) # Shuffle a smaller buffer for speed
    .map(parse_image, num_parallel_calls=AUTOTUNE)
    .map(lambda image, label: augmentor(image, label, image_size=image_size), num_parallel_calls=AUTOTUNE)
    # ✅ UPDATE: Pass the model_name to the preprocessing function
    .map(lambda image, label: preprocess_data(image, label, MODEL_NAME), num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)
print("--- Pipeline built. Taking one batch... ---")

# Get one batch of data from the final pipeline
image_batch, label_batch = next(iter(train_ds))

# --- Analyze the Batch ---
print("\n--- Data Batch Analysis ---")
print(f"Batch shape: {image_batch.shape}")
print(f"Data type: {image_batch.dtype}")
min_val = tf.reduce_min(image_batch)
max_val = tf.reduce_max(image_batch)
mean_val = tf.reduce_mean(image_batch)
print(f"Min pixel value in batch: {min_val:.4f}")
print(f"Max pixel value in batch: {max_val:.4f}")
print(f"Mean pixel value in batch: {mean_val:.4f}")
print("---------------------------\n")


# --- Visualize the Batch ---
def rescale_for_plotting(image):
    # This is a simple min-max normalization for viewing purposes
    return (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

plt.figure(figsize=(15, 15))
plt.suptitle(f"Images As Seen by {MODEL_NAME} (After Preprocessing)")
for i in range(min(16, batch_size)): # Show up to 16 images
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(rescale_for_plotting(image_batch[i]))
    plt.title(class_names[label_batch[i]], fontsize=8)
    plt.axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()