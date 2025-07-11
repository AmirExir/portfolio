import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# Load model artifacts
try:
    model = joblib.load("fault_model.pkl")
    scaler = joblib.load("scaler.pkl")
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

        # Make predictions
        predicted_bits = model.predict(X_scaled)

        # Convert each prediction into a 4-bit string like "0110"
        predicted_faults = [''.join(map(str, row.astype(int))) for row in predicted_bits]

        st.subheader("🔍 Predicted Fault Types:")
        st.write(pd.DataFrame(predicted_faults, columns=["Fault Code"]))

        # Download results button
        result_df = pd.DataFrame(predicted_faults, columns=["Fault Code"])
        st.download_button("📥 Download Results CSV", result_df.to_csv(index=False), "predictions.csv", "text/csv")

        # Optional: Hardcoded accuracy plot
        st.subheader("📊 Model Accuracy Comparison (from training script)")
        fig, ax = plt.subplots()
        model_names = ['LogReg', 'RandomForest', 'SVM', 'MLP', 'XGBoost']
        accuracies = [0.90, 0.88, 0.81, 0.86, 0.83]
        ax.bar(model_names, accuracies, color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison: Fault Type Classification")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong: {e}")