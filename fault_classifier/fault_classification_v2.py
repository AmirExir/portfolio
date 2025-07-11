# fault_classification_v2.py

import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split  # ❌ OLD — Not splitting anymore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib
import json

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    print("XGBoost not installed. Skipping XGBClassifier.")

# ✅ Load full training dataset
df_train = pd.read_csv("classData.csv")

# Preprocessing
df_train["fault_type"] = df_train[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
print(df_train["fault_type"].value_counts())
X_train = df_train[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]]
y_train = df_train["fault_type"]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
print("Label Mapping:", label_encoder.classes_)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ✅ Load test set (real-world)
df_test = pd.read_csv("detect_dataset.csv")
X_test = df_test[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]]

# Optional: load true labels if available
has_labels = "fault_type" in df_test.columns
if has_labels:
    df_test["fault_type"] = df_test[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
    y_test = label_encoder.transform(df_test["fault_type"])

X_test_scaled = scaler.transform(X_test)

# Models
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

for name, model in models.items():
    print(f"\n🔍 Training: {name}")
    if name in ["Logistic Regression", "SVM (RBF Kernel)", "MLP (Neural Net)"]:
        model.fit(X_train_scaled, y_train_encoded)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)

    if has_labels:
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"✅ Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    else:
        print("⚠️ No labels available in detect_dataset.csv — skipping accuracy.")

    if has_labels and acc > best_acc:
        best_acc = acc
        best_model = model

# If no labeled test set, just pick last model as "best"
if not best_model:
    best_model = list(models.values())[0]

print("✅ Final best model saved:", type(best_model))

# Save artifacts
joblib.dump(best_model, "fault_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Saved model, scaler, and label encoder to .pkl files")

# Save model results
with open("model_accuracies.json", "w") as f:
    json.dump(results, f, indent=4)

# Save predictions
y_pred_labels = label_encoder.inverse_transform(y_pred)
df_test["Predicted"] = y_pred_labels
df_test.to_csv("predictions_on_detect_dataset.csv", index=False)
print("✅ Predictions saved to predictions_on_detect_dataset.csv")

# Accuracy chart
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel("Accuracy")
plt.title("Model Comparison: Fault Type Classification")
plt.xticks(rotation=15)
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("images/fault_accuracy_comparison.png")
plt.show()