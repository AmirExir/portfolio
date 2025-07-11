import streamlit as st
import pandas as pd
import joblib

# Load saved model components
model = joblib.load("fault_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="⚡ Fault Classifier", page_icon="⚡")
st.title("⚡ Power System Fault Classifier by Amir Exir")
st.write("Upload a CSV with columns: Ia, Ib, Ic, Va, Vb, Vc")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(df.head())

        # Preprocess and predict
        scaled = scaler.transform(df)
        predictions = model.predict(scaled)
        predicted_labels = label_encoder.inverse_transform(predictions)

        st.subheader("🔍 Predicted Fault Types:")
        st.write(predicted_labels)

        # Optional: Add download button
        result_df = df.copy()
        result_df["Predicted Fault Type"] = predicted_labels
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Something went wrong: {e}")