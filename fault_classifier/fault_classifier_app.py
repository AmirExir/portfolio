import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ✅ Streamlit config — must go FIRST
st.set_page_config(page_title="Power Fault Classifier", layout="centered")

# App Header
st.title("⚡ Power System Fault Classifier by Amir Exir")
st.write("Upload a CSV with columns: `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc` — optionally `G`, `C`, `B`, `A` for true labels")

# Load model artifacts
try:
    model = joblib.load("fault_classifier/fault_model.pkl")
    scaler = joblib.load("fault_classifier/scaler.pkl")
    label_encoder = joblib.load("fault_classifier/label_encoder.pkl")

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

        # === COMPARISON PLOT: Prefold vs K-Fold ===
        try:
            with open("fault_classifier/model_accuracies_prefold.json", "r") as f1, open("model_accuracies.json", "r") as f2:
                pre_fold = json.load(f1)
                post_fold = json.load(f2)

            st.subheader("📊 Accuracy Comparison: Train/Test Split vs 5-Fold Cross-Validation")

            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(pre_fold.keys())
            x = range(len(models))

            pre_values = [pre_fold[m] for m in models]
            post_values = [post_fold.get(m, 0) for m in models]

            ax.bar([i - 0.2 for i in x], pre_values, width=0.4, label="Train/Test Split", color="skyblue")
            ax.bar([i + 0.2 for i in x], post_values, width=0.4, label="5-Fold CV", color="orange")

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha='right')
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Accuracy: Train/Test Split vs 5-Fold CV")
            ax.legend()

            for i, (p, q) in enumerate(zip(pre_values, post_values)):
                ax.text(i - 0.2, p + 0.002, f"{p:.3f}", ha="center", fontsize=9)
                ax.text(i + 0.2, q + 0.002, f"{q:.3f}", ha="center", fontsize=9)

            st.pyplot(fig)

            st.subheader("🧾 Accuracy Scores")
            for model in models:
                st.write(f"🔹 **{model}** → Pre-Fold: `{pre_fold[model]:.4f}`, 5-Fold CV: `{post_fold[model]:.4f}`")

        except Exception as e:
            st.warning(f"⚠️ Accuracy comparison failed to load: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")