# src/evaluate.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from dataset import test_df, AUTOTUNE, class_names
from helpers import decode_test, view_prediction, view_wrong_prediction, training_history

# --- ✅ 1. CHOOSE WHICH MODEL TO EVALUATE ---
MODEL_NAME = "ResNet50"  # Options: "ResNet50", "EfficientNetB0"
USE_ATTENTION = True     # Options: True, False

# --- Load Model and History ---
attention_str = "with_attention" if USE_ATTENTION else "without_attention"
model_filename = f"best_model_{MODEL_NAME}_{attention_str}.h5"
history_filename = f"history_{MODEL_NAME}_{attention_str}.npy"

model_path = os.path.join("outputs", "models", model_filename)
history_path = os.path.join("outputs", history_filename)

assert os.path.exists(model_path), f"❌ Model file not found at {model_path}. Run train.py for this configuration first."
assert os.path.exists(history_path), f"❌ History file not found at {history_path}. Run train.py for this configuration first."

model = tf.keras.models.load_model(model_path)
print(f"✅ Loaded trained model from {model_path}")
history_data = np.load(history_path, allow_pickle=True).item()
print(f"✅ Loaded training history from {history_path}")

# --- Prepare Test Dataset ---
# ✅ 2. UPDATE the pipeline to use the correct preprocessing for the loaded model
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_df.path)
    .map(lambda path: decode_test(path, model_name=MODEL_NAME), num_parallel_calls=AUTOTUNE)
    .batch(32)
)
test_img_batches = list(test_ds.as_numpy_iterator())
test_img = np.concatenate(test_img_batches, axis=0)
test_index = test_df.label_int.values
test_label = test_df.label.values

# --- Model Predictions ---
test_pred_probs = model.predict(test_img)
# ✅ 3. UPDATE the prediction logic for multi-class
pred_index = np.argmax(test_pred_probs, axis=1) # Get the index of the highest probability
pred_label = np.array(class_names)[pred_index]

# --- Reports and Plots ---
print("\n📈 Displaying Training History...")
training_history(history_data)

print("\n📊 Classification Report:")
print(classification_report(test_index, pred_index, target_names=class_names, zero_division=0))
print("accuracy_score :", accuracy_score(test_index, pred_index))

# Show the Confusion Matrix (now 8x8)
cm = confusion_matrix(test_index, pred_index)
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix - {MODEL_NAME} {'with Attention' if USE_ATTENTION else 'without Attention'}")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Save Prediction CSVs ---
prediction_df = pd.DataFrame({
    'filename': test_df.filename.values,
    'actual': test_df.label.values,
    'prediction': pred_label,
    'path': test_df.path.values,
})
os.makedirs(os.path.join("outputs"), exist_ok=True)
csv_filename = f"predictions_{MODEL_NAME}_{attention_str}.csv"
prediction_df.to_csv(os.path.join("outputs", csv_filename), index=False)
wrong_df = prediction_df[prediction_df.actual != prediction_df.prediction].reset_index(drop=True)
wrong_df.to_csv(os.path.join("outputs", f"wrong_{csv_filename}"), index=False)
print(f"✅ Saved predictions to outputs/{csv_filename}")

# --- Show Prediction Visualizations ---
# Note: For 8 classes, a grid of 30 might be too crowded. This can be adjusted.
print("🔍 Showing sample predictions...")
view_prediction(test_img=test_img, pred_label=pred_label)
print("🔍 Showing wrongly classified predictions...")
if not wrong_df.empty:
    view_wrong_prediction(wrong_df, model_name=MODEL_NAME)
else:
    print("🎉 No wrong predictions!")