import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
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

# === MODELS ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF Kernel)": SVC(),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}
if has_xgb:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# === PREFOLD: Train/Test Split Accuracy ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
prefold_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prefold_results[name] = acc
    print(f"📊 {name} Pre-Fold Accuracy: {acc:.4f}")

with open("model_accuracies_prefold.json", "w") as f:
    json.dump(prefold_results, f, indent=4)

# === 5-FOLD CROSS VALIDATION ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = {}
results = {}
best_model = None
best_acc = 0

for name, model in models.items():
    print(f"\n🔍 Cross-validating: {name}")
    acc_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_scaled), 1):
        X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
        y_train_fold, y_val_fold = y_encoded[train_index], y_encoded[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred_fold)
        acc_scores.append(acc)
        print(f"  Fold {fold}: Accuracy = {acc:.4f}")

    avg_acc = np.mean(acc_scores)
    results[name] = avg_acc
    fold_accuracies[name] = acc_scores
    print(f"✅ Average CV Accuracy: {avg_acc:.4f}")

    if avg_acc > best_acc:
        best_acc = avg_acc
        best_model = model

# === Save post-fold results ===
with open("model_accuracies.json", "w") as f:
    json.dump(results, f, indent=4)

# === Retrain Best Model on Full Data ===
best_model.fit(X_scaled, y_encoded)

# === SAVE FINAL ARTIFACTS ===
joblib.dump(best_model, "fault_classifier/fault_model.pkl")
joblib.dump(scaler, "fault_classifier/scaler.pkl")
joblib.dump(label_encoder, "fault_classifier/label_encoder.pkl")

print("✅ Saved model, scaler, encoder, and accuracy")

# === CREATE OUTPUT FOLDER ===
os.makedirs("images", exist_ok=True)

# === BOX PLOT: Per-Fold Accuracies ===
plt.figure(figsize=(10, 6))
plt.boxplot(fold_accuracies.values(), labels=fold_accuracies.keys())
plt.ylabel("Accuracy")
plt.title("Per-Fold Accuracy (5-Fold CV)")
plt.xticks(rotation=15)
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("images/foldwise_accuracy_boxplot.png")
plt.show()

# === BAR CHART: Average Accuracy ===
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel("Avg CV Accuracy")
plt.title("Model Accuracy (5-Fold Cross-Validation)")
plt.xticks(rotation=15)
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("images/fault_accuracy_comparison.png")
plt.show()

# === PREDICT ON DETECT DATASET ===
if os.path.exists("detect_dataset.csv"):
    df_detect = pd.read_csv("detect_dataset.csv")
    X_detect = df_detect[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]]
    X_detect_scaled = scaler.transform(X_detect)

    y_detect_pred = best_model.predict(X_detect_scaled)
    y_detect_labels = label_encoder.inverse_transform(y_detect_pred)

    df_detect["Predicted"] = y_detect_labels
    df_detect.to_csv("predictions_on_detect_dataset.csv", index=False)
    print("✅ Predictions on detect_dataset saved.")