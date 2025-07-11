# fault_classification_v2.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    print("XGBoost not installed. Skipping XGBClassifier.")

# ✅ Load CSV from disk instead of Colab upload
df = pd.read_csv("classData.csv")

# Preprocessing
df["fault_type"] = df[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
print(df["fault_type"].value_counts())  # ← This line shows label distribution
X = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]]
y = df["fault_type"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Label Mapping:", label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
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
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    if acc > best_acc:
        best_acc = acc
        best_model = model
print("✅ Final best model saved:", type(best_model))        

# Save the best model and scaler for Streamlit
joblib.dump(best_model, "fault_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# ✅ Save model accuracy results to JSON for Streamlit
import json
with open("model_accuracies.json", "w") as f:
    json.dump(results, f, indent=4)

# Accuracy bar chart
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel("Accuracy")
plt.title("Model Comparison: Fault Type Classification")
plt.xticks(rotation=15)
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("images/fault_accuracy_comparison.png")  # Optional
plt.show()