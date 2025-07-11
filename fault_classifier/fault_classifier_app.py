import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# ✅ Streamlit config — must go FIRST
st.set_page_config(page_title="Power Fault Classifier", layout="centered")

# App Header
st.title("⚡ Power System Fault Classifier by Amir Exir")
st.write("Upload a CSV with columns: `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc` — optionally `G`, `C`, `B`, `A` for true labels")

# Load model artifacts
try:
    model = joblib.load("fault_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    st.subheader("🧠 Debug: Model & Label Info")
    if hasattr(model, "classes_"):
        st.write("Model classes:", model.classes_)
    st.write("Label encoder classes:", label_encoder.classes_)

except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}")
    st.stop()

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload test CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.write(df.head())

        feature_cols = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
        if not all(col in df.columns for col in feature_cols):
            st.error("❌ Required columns missing.")
            st.stop()

        X = df[feature_cols]
        st.subheader("🔎 Features Before Scaling")
        st.write(X.head())

        X_scaled = scaler.transform(X)
        st.subheader("📐 Features After Scaling")
        st.write(pd.DataFrame(X_scaled, columns=feature_cols).head())

        # Prediction
        predicted_faults = model.predict(X_scaled)
        df_predictions = pd.DataFrame(predicted_faults, columns=["Fault Code"])

        fault_type_names = dict(enumerate(label_encoder.classes_))
        df_predictions["Fault String"] = df_predictions["Fault Code"].map(fault_type_names)

        st.subheader("🧪 Prediction Debug Info")
        st.write("🧭 Label decoder map:", fault_type_names)
        st.write("🔢 Predicted Class Indices:", predicted_faults[:10])
        st.write("🔤 Fault String Counts:")
        st.write(df_predictions["Fault String"].value_counts())

        # ✅ Optional ground truth comparison
        if all(col in df.columns for col in ["G", "C", "B", "A"]):
            df["True Fault"] = df[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)

            if set(df["True Fault"]).issubset(set(label_encoder.classes_)):
                df_predictions["True Fault"] = df["True Fault"]

                st.subheader("✅ Ground Truth vs Prediction")
                st.dataframe(df_predictions[["True Fault", "Fault String"]])

                y_true = label_encoder.transform(df_predictions["True Fault"])
                y_pred = df_predictions["Fault Code"]

                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                            xticklabels=label_encoder.classes_,
                            yticklabels=label_encoder.classes_)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                ax.set_title("📉 Confusion Matrix")
                st.pyplot(fig)

        # Final results
        st.subheader("🔍 Predicted Fault Types")
        st.dataframe(df_predictions)

        # Download
        st.download_button("📥 Download Results", df_predictions.to_csv(index=False), "predictions.csv", "text/csv")

        
        # Accuracy Chart with Precision + Clear Labels
        with open("model_accuracies.json", "r") as f:
            model_accuracies = json.load(f)

        st.subheader("📊 Model Accuracy Comparison")

        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(model_accuracies.keys())
        scores = list(model_accuracies.values())

        min_acc = min(scores)
        ax.set_ylim(min_acc - 0.01, 1.0)

        bars = ax.bar(models, scores, color='skyblue')
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy (Validation on classData)")

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        st.pyplot(fig)

        st.subheader("🧾 Accuracy Scores")
        for model, acc in model_accuracies.items():
            st.write(f"🔹 **{model}**: `{acc:.4f}`")