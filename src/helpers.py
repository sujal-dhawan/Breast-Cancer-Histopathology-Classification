# src/helpers.py

import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- Image Parsing and Preprocessing ---
def parse_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return img, label

def resize_image(image, label, image_size=224):
    img = tf.cast(image, tf.float32)
    img = tf.image.resize(img, [image_size, image_size])
    return img, label

def preprocess_data(image, label):
    image = preprocess_input(image)
    return image, label

def decode_test(path, image_size=224):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img, _ = resize_image(img, None, image_size)
    img = preprocess_input(img)
    return img

# --- Augmentation ---
def aug_fn(image, image_size=224):
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5, limit=15),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.1, 0.1)),
        A.RandomResizedCrop(p=0.8, height=image_size, width=image_size, scale=(0.8, 1.0)),
        A.Blur(p=0.3, blur_limit=(1, 2)),
    ])
    aug_data = transforms(image=image)
    return aug_data["image"]

def augmentor(image, label, image_size=224):
    def apply_aug(img):
        return aug_fn(img.numpy(), image_size)
    aug_img = tf.py_function(func=apply_aug, inp=[image], Tout=tf.uint8)
    aug_img.set_shape([image_size, image_size, 3])
    aug_img, label = resize_image(aug_img, label, image_size)
    return aug_img, label

# --- Visualization ---

def _rescale_for_display(image):
    """Rescales a preprocessed image from any range to the [0, 1] range for plotting."""
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def view_image(ds, class_names, col=8, row=2, size=(25, 7)):
    plt.figure(figsize=size)
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    for images, labels in ds.take(1):
        for i in range(min(col * row, len(images))):
            ax = plt.subplot(row, col, i + 1)
            ax.imshow(_rescale_for_display(images[i].numpy()))
            ax.set_title(class_names[labels[i].numpy()])
            ax.axis("off")
    plt.tight_layout()
    plt.show()

def training_history(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs_range = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    ax1.plot(epochs_range, acc, label='Training Accuracy')
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy')
    ax1.legend(loc='lower right')
    ax1.set_title('Training and Validation Accuracy')

    ax2.plot(epochs_range, loss, label='Training Loss')
    ax2.plot(epochs_range, val_loss, label='Validation Loss')
    ax2.legend(loc='upper right')
    ax2.set_title('Training and Validation Loss')
    plt.show()

def view_prediction(test_img=None, pred_label=None, max_show=30):
    if test_img is None or pred_label is None:
        print("⚠️ view_prediction called without inputs")
        return
    plt.figure(figsize=(25, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    for i in range(min(max_show, len(test_img))):
        ax = plt.subplot(3, 10, i + 1)
        ax.imshow(_rescale_for_display(test_img[i]))
        ax.set_title(str(pred_label[i]), fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def view_wrong_prediction(df, image_size=224):
    if df.empty:
        print("🎉 No wrong predictions to display")
        return
    num_images = len(df)
    cols = min(num_images, 8)
    rows = (num_images - 1) // cols + 1
    plt.figure(figsize=(cols * 3, rows * 4))

    for i in range(num_images):
        img = decode_test(df.path.iloc[i], image_size)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(_rescale_for_display(img.numpy()))
        ax.set_title(f"Pred: {df.prediction.iloc[i]}\nActual: {df.actual.iloc[i]}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()