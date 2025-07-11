import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
        
        # Preprocess and predict
        features = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
        df_features = df[features]
        scaled = scaler.transform(df_features)
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
# Display model accuracy chart (optional - static, from training phase)
st.subheader("📊 Model Accuracy Comparison (from training script)")

# Hardcoded results — you can later make it dynamic
model_names = ['Logistic Regression', 'Random Forest', 'SVM (RBF)', 'MLP', 'XGBoost']
accuracies = [0.36, 0.88, 0.81, 0.86, 0.83]  # Replace with your actual results

fig, ax = plt.subplots()
ax.bar(model_names, accuracies, color='skyblue')
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Model Comparison: Fault Type Classification")
plt.xticks(rotation=15)
st.pyplot(fig)