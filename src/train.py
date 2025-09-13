# src/train.py

import os
import tensorflow as tf
import numpy as np

# Keras components
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

# import dataset and helpers
from dataset import train_df, valid_df, class_names, AUTOTUNE
from helpers import parse_image, augmentor, resize_image, preprocess_data

# --- Training Config ---
image_size = 224
batch_size = 64
epochs = 12 # Simplified to just the initial epochs

print(f"Training samples: {len(train_df)}")
print(f"Image size: {image_size}, Batch size: {batch_size}, Epochs: {epochs}")

# --- Create TensorFlow Datasets ---
train_loader = tf.data.Dataset.from_tensor_slices((train_df.path, train_df.label_int))
valid_loader = tf.data.Dataset.from_tensor_slices((valid_df.path, valid_df.label_int))

train_ds = (
    train_loader.shuffle(len(train_df))
    .map(parse_image, num_parallel_calls=AUTOTUNE)
    .map(lambda image, label: augmentor(image, label, image_size=image_size), num_parallel_calls=AUTOTUNE)
    .map(preprocess_data, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    valid_loader.shuffle(len(valid_df))
    .map(parse_image, num_parallel_calls=AUTOTUNE)
    .map(lambda image, label: resize_image(image, label, image_size=image_size), num_parallel_calls=AUTOTUNE)
    .map(preprocess_data, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

print("Datasets ready!")

# --- Construct neural network ---
tf.keras.backend.clear_session()

def build_network(input_size):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_network(image_size)

model_save_path = os.path.join("outputs", "models", "best_model.h5")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')

METRICS = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]

# --- Model Training ---
print("--- Starting Model Training ---")
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=METRICS
)

history = model.fit(
    train_ds,
    epochs=epochs,
    verbose=1,
    callbacks=[checkpoint_cb],
    validation_data=valid_ds,
)

# --- Save Training History ---
print("✅ Training complete. Saving training history to 'outputs/training_history.npy'...")
np.save('outputs/training_history.npy', history.history)

print("✅ All tasks complete.")