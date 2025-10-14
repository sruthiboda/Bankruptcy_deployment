import streamlit as st
import numpy as np
import joblib

# Load model and scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit page config
st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

# Title and description
st.markdown("<h1 style='text-align:center;color:white;'>🏦 Bankruptcy Prediction</h1>", unsafe_allow_html=True)
st.write("Fill in the 64 financial features below to predict if a company is Bankrupt or Not.")

# 64 input fields arranged in 8 rows × 8 columns
features = []
cols = st.columns(8)

for i in range(64):
    with cols[i % 8]:
        val = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f")
        features.append(val)

# Prediction button
if st.button("Predict"):
    try:
        # Convert inputs to array
        features_array = np.array(features).reshape(1, -1)

        # Scale the inputs
        features_scaled = scaler.transform(features_array)

        # Predict probabilities
        xgb_pred_proba = xgb_model.predict_proba(features_scaled)[:, 1]
        prediction = "Bankrupt" if xgb_pred_proba[0] > 0.5 else "Not Bankrupt"

        # Show result
        if prediction == "Bankrupt":
            st.error(f"Prediction: **{prediction}** 💔 (Probability: {xgb_pred_proba[0]:.2%})")
        else:
            st.success(f"Prediction: **{prediction}** ✅ (Probability: {xgb_pred_proba[0]:.2%})")

    except Exception as e:
        st.error(f"Error: {str(e)}")
