# src/evaluate.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from dataset import test_df, AUTOTUNE, class_names
# ✅ 1. Import the training_history function
from helpers import decode_test, view_prediction, view_wrong_prediction, training_history


# --- Load Model and History ---
model_path = os.path.join("outputs", "models", "best_model.h5")
history_path = os.path.join("outputs", "training_history.npy")

assert os.path.exists(model_path), f"❌ Model file not found at {model_path}. Run train.py first."
# ✅ 2. Add a check for the new history file
assert os.path.exists(history_path), f"❌ History file not found at {history_path}. Run train.py first."

model = tf.keras.models.load_model(model_path)
print(f"✅ Loaded trained model from {model_path}")

history_data = np.load(history_path, allow_pickle=True).item()
print(f"✅ Loaded training history from {history_path}")


# --- Prepare Test Dataset ---
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_df.path)
    .map(decode_test, num_parallel_calls=AUTOTUNE)
    .batch(32)
)
test_img_batches = list(test_ds.as_numpy_iterator())
test_img = np.concatenate(test_img_batches, axis=0)
test_index = test_df.label_int.values
test_label = test_df.label.values


# --- Model Predictions ---
test_pred = model.predict(test_img)
pred_index = (test_pred > 0.5).astype('uint8')
pred_label = np.array(class_names)[pred_index.flatten()]


# --- Reports and Plots ---

# ✅ 3. Show the Training History plot first
print("\n📈 Displaying Training History...")
training_history(history_data)

# Show the Classification Report
print("\n📊 Classification Report:")
print(classification_report(test_index, pred_index, target_names=class_names, zero_division=0))
print("f1_score       :", f1_score(test_index, pred_index, average='micro'))
print("accuracy_score :", accuracy_score(test_index, pred_index))

# Show the Confusion Matrix
cm = confusion_matrix(test_label, pred_label, labels=class_names)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# --- Save Prediction CSVs ---
prediction_df = pd.DataFrame({
    'filename': test_df.filename.values,
    'actual': test_df.label.values,
    'prediction': pred_label,
    'path': test_df.path.values,
})
os.makedirs(os.path.join("outputs"), exist_ok=True)
prediction_df.to_csv(os.path.join("outputs", "predictions.csv"), index=False)
wrong_df = prediction_df[prediction_df.actual != prediction_df.prediction].reset_index(drop=True)
wrong_df.to_csv(os.path.join("outputs", "wrong_predictions.csv"), index=False)
print(f"✅ Saved predictions to outputs/predictions.csv")
print(f"✅ Saved wrong predictions to outputs/wrong_predictions.csv")


# --- Show Prediction Visualizations ---
print("🔍 Showing first 30 predictions...")
view_prediction(test_img=tf.convert_to_tensor(test_img), pred_label=pred_label)
print("🔍 Showing wrongly classified predictions...")
if not wrong_df.empty:
    view_wrong_prediction(wrong_df)
else:
    print("🎉 No wrong predictions!")