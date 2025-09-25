# src/train.py

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0

# import dataset and helpers
from dataset import train_df, valid_df, class_names, AUTOTUNE
from helpers import (
    parse_image, augmentor, resize_image, preprocess_data,
    SqueezeExciteBlock, training_history
)

# --- CHOOSE YOUR MODEL AND SETTINGS ---
MODEL_NAME = "ResNet50"  # Options: "ResNet50", "EfficientNetB0"
USE_ATTENTION = True  # Options: True, False
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# --- Create TensorFlow Datasets ---
train_loader = tf.data.Dataset.from_tensor_slices((train_df.path, train_df.label_int))
valid_loader = tf.data.Dataset.from_tensor_slices((valid_df.path, valid_df.label_int))


# ✅ DEFINE the one-hot encoding function
def one_hot_label(image, label):
    return image, tf.one_hot(label, depth=len(class_names))


train_ds = (
    train_loader.shuffle(len(train_df))
    .map(parse_image, num_parallel_calls=AUTOTUNE)
    .map(lambda image, label: augmentor(image, label, image_size=IMAGE_SIZE), num_parallel_calls=AUTOTUNE)
    .map(lambda image, label: preprocess_data(image, label, MODEL_NAME), num_parallel_calls=AUTOTUNE)
    .map(one_hot_label, num_parallel_calls=AUTOTUNE)  # ✅ ADD this step to one-hot encode labels
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    valid_loader.shuffle(len(valid_df))
    .map(parse_image, num_parallel_calls=AUTOTUNE)
    .map(lambda image, label: resize_image(image, label, image_size=IMAGE_SIZE), num_parallel_calls=AUTOTUNE)
    .map(lambda image, label: preprocess_data(image, label, MODEL_NAME), num_parallel_calls=AUTOTUNE)
    .map(one_hot_label, num_parallel_calls=AUTOTUNE)  # ✅ ADD this step here as well
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("✅ Datasets ready for 8-class classification!")


# --- (The rest of the file remains exactly the same) ---

def build_network(model_name, use_attention, input_size=224, num_classes=8):
    print(f"Building model: {model_name} with Attention: {use_attention}")

    if model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    else:
        raise ValueError("Unsupported model name")

    base_model.trainable = False

    x = base_model.output
    if use_attention:
        x = SqueezeExciteBlock(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


model = build_network(MODEL_NAME, USE_ATTENTION, input_size=IMAGE_SIZE, num_classes=len(class_names))
model.summary()

# --- Model Training ---
attention_str = "with_attention" if USE_ATTENTION else "without_attention"
model_save_path = os.path.join("outputs", "models", f"best_model_{MODEL_NAME}_{attention_str}.h5")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy',
                                                   mode='max')

METRICS = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),
    loss='categorical_crossentropy',
    metrics=METRICS
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint_cb],
    validation_data=valid_ds,
)

# --- Save Training History ---
history_save_path = os.path.join("outputs", f"history_{MODEL_NAME}_{attention_str}.npy")
print(f"✅ Training complete. Saving training history to '{history_save_path}'...")
np.save(history_save_path, history.history)

print("✅ All tasks complete.")
training_history(history.history)