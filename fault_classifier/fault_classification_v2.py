import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib
import json
import os

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    print("⚠️ XGBoost not installed.")

# === LOAD & PREPROCESS TRAINING DATA ===
df = pd.read_csv("classData.csv")
df["fault_type"] = df[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
X = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]]
y = df["fault_type"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === EVALUATE: Split for validation only ===
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# === MODELS ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF Kernel)": SVC(),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}
if has_xgb:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

results = {}
best_model = None
best_acc = 0

# === VALIDATE MODELS ===
for name, model in models.items():
    print(f"\n🔍 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    if acc > best_acc:
        best_acc = acc
        best_model = model

# === Retrain Best Model on Full Data ===
best_model.fit(X_scaled, y_encoded)

# === SAVE FINAL ARTIFACTS ===
joblib.dump(best_model, "fault_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

with open("model_accuracies.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Saved model, scaler, encoder, and accuracy")

# === CONFUSION MATRIX (Validation Data) ===
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45, cmap='viridis', colorbar=True)
plt.title("Validation Confusion Matrix (classData split)")
plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/confusion_matrix_classdata.png")
plt.show()

# === Predict on REAL detect_dataset.csv ===
df_detect = pd.read_csv("detect_dataset.csv")
X_detect = df_detect[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]]
X_detect_scaled = scaler.transform(X_detect)

y_detect_pred = best_model.predict(X_detect_scaled)
y_detect_labels = label_encoder.inverse_transform(y_detect_pred)

df_detect["Predicted"] = y_detect_labels
df_detect.to_csv("predictions_on_detect_dataset.csv", index=False)
print("✅ Predictions on detect_dataset saved.")

# === Plot Model Accuracies ===
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel("Accuracy")
plt.title("Model Accuracy (Validation on classData)")
plt.xticks(rotation=15)
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("images/fault_accuracy_comparison.png")
plt.show()