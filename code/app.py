import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import numpy as np

st.title("Customer Churn Prediction Using ANN")

# Load scaler and model once
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("ANN_scaler.pkl")
    model = load_model("churn_model.h5")
    return scaler, model

scaler, model = load_artifacts()

# Load and preprocess dataset for visualizations and UI
df = pd.read_csv("customer_churn.csv")
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Sidebar input collection
st.sidebar.header("Input Customer Details")
input_data = {}
for col in df.drop('Churn', axis=1).columns:
    unique_vals = df[col].unique()
    if len(unique_vals) == 2 and sorted(unique_vals) == [0,1]:
        input_data[col] = st.sidebar.selectbox(f"{col}", options=[0, 1])
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)

input_df = pd.DataFrame([input_data])

# When predict button clicked
if st.button("Predict Churn"):
    # Scale input using saved scaler
    input_scaled = scaler.transform(input_df)

    # Predict with saved model
    pred_prob = model.predict(input_scaled)[0][0]
    st.write(f"Churn Probability: **{pred_prob:.2f}**")

    if pred_prob < 0.5:
        st.success("Customer is likely to stay.")
    else:
        st.error("Customer is likely to churn.")

# Visualizations
st.header("Data Visualizations")

# Correlation heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
st.pyplot(plt.gcf())  # Use gcf() to avoid matplotlib warnings

# Boxplot of MonthlyCharges by Churn
st.subheader("Boxplot of MonthlyCharges by Churn")
plt.figure(figsize=(8, 5))
original_df = pd.read_csv("customer_churn.csv")
sns.boxplot(x='Churn', y='MonthlyCharges', data=original_df)
st.pyplot(plt.gcf())
