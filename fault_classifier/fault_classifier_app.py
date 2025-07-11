import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import json

# Load model artifacts
try:
    model = joblib.load("fault_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()

# Streamlit app UI
st.set_page_config(page_title="Power Fault Classifier", layout="centered")
st.title("⚡ Power System Fault Classifier by Amir Exir")
st.write("Upload a CSV with columns: `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Show preview
        st.subheader("📄 Uploaded Data Preview:")
        st.write(df.head())

        # Features
        feature_cols = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
        if not all(col in df.columns for col in feature_cols):
            st.error("Missing required columns in uploaded CSV.")
            st.stop()

        X = df[feature_cols]
        X_scaled = scaler.transform(X)

        # Predict
        predicted_faults = model.predict(X_scaled)

        # Decode fault labels using label_encoder
        df_predictions = pd.DataFrame(predicted_faults, columns=["Fault Code"])
        fault_type_names = dict(enumerate(label_encoder.classes_))
        df_predictions["Fault String"] = df_predictions["Fault Code"].map(fault_type_names)

        # Show results
        st.subheader("🔍 Predicted Fault Types:")
        st.dataframe(df_predictions)

        # Download button
        st.download_button("📥 Download Results CSV", df_predictions.to_csv(index=False), "predictions.csv", "text/csv")

        # Accuracy chart (loaded from training)
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
        st.error(f"Something went wrong: {e}")