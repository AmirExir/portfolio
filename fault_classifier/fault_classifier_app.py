import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import json

# ✅ Streamlit config — must go FIRST
st.set_page_config(page_title="Power Fault Classifier", layout="centered")

# App Header
st.title("⚡ Power System Fault Classifier by Amir Exir")
st.write("Upload a CSV with columns: `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`")

# Load model artifacts
try:
    model = joblib.load("fault_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # ✅ Check model and label encoder class alignment
    st.subheader("🧠 Debug: Model Classes and Label Encoder")
    if hasattr(model, "classes_"):
        st.write("Model classes:", model.classes_)
    else:
        st.write("Model does not expose `.classes_`")

    st.write("Label encoder classes:", label_encoder.classes_)

except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}")
    st.stop()

# Upload CSV
uploaded_file = st.file_uploader("📂 Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Show preview
        st.subheader("📄 Uploaded Data Preview:")
        st.write(df.head())

        # Feature check
        feature_cols = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
        if not all(col in df.columns for col in feature_cols):
            st.error("❌ Missing required columns in uploaded CSV.")
            st.stop()

        X = df[feature_cols]
        st.subheader("🔎 Input Features (Before Scaling)")
        st.write(X.head())

        # Scale inputs
        X_scaled = scaler.transform(X)
        st.subheader("📐 Input Features (After Scaling)")
        st.write(pd.DataFrame(X_scaled, columns=feature_cols).head())

        # Predict
        predicted_faults = model.predict(X_scaled)
        df_predictions = pd.DataFrame(predicted_faults, columns=["Fault Code"])

        # Decode labels
        fault_type_names = dict(enumerate(label_encoder.classes_))
        df_predictions["Fault String"] = df_predictions["Fault Code"].map(fault_type_names)

        # Show results
        st.subheader("🔍 Predicted Fault Types:")
        st.dataframe(df_predictions)

        # Download results
        st.download_button("📥 Download Results CSV", df_predictions.to_csv(index=False), "predictions.csv", "text/csv")

        # Accuracy chart
        try:
            with open("model_accuracies.json", "r") as f:
                model_accuracies = json.load(f)

            st.subheader("📊 Model Accuracy Comparison (from training script)")
            fig, ax = plt.subplots()
            ax.bar(model_accuracies.keys(), model_accuracies.values(), color='skyblue')
            ax.set_ylim(0, 1)
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Comparison: Fault Type Classification")
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"⚠️ Could not load accuracy chart: {e}")

    except Exception as e:
        st.error(f"💥 Something went wrong: {e}")