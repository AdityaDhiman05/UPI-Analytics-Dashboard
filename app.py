# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# ------------------------------
# Title
# ------------------------------
st.title("📊 UPI Analytics Dashboard")
st.markdown("Explore UPI transaction trends and model predictions")

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("upi_train_prepared.csv")
    return df

df = load_data()

# ------------------------------
# Load trained model
# ------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_random_forest_model.pkl")
    return model

best_model = load_model()

# ------------------------------
# KPIs
# ------------------------------
st.subheader("Key Metrics")
total_volume = df["Volume_Mn"].sum()
avg_volume = df["Volume_Mn"].mean()
max_volume = df["Volume_Mn"].max()
min_volume = df["Volume_Mn"].min()
total_banks = df["Banks_Live_Scaled"].max()

st.write(f"**Total UPI Volume (Mn):** {total_volume:.2f}")
st.write(f"**Average Volume (Mn):** {avg_volume:.2f}")
st.write(f"**Max Volume (Mn):** {max_volume:.2f}")
st.write(f"**Min Volume (Mn):** {min_volume:.2f}")
st.write(f"**Banks Live (Scaled):** {total_banks:.2f}")

# ------------------------------
# Predictions
# ------------------------------
st.subheader("Predicted vs Actual UPI Volume")
X = df[["Banks_Live_Scaled"]]
y = df["Volume_Mn"]
y_pred = best_model.predict(X)

fig = px.scatter(x=y, y=y_pred,
                 labels={"x": "Actual Volume (Mn)", "y": "Predicted Volume (Mn)"},
                 title="Predicted vs Actual Volume")
fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
              line=dict(color="black", dash="dash"))
st.plotly_chart(fig)

# ------------------------------
# Banks Live vs Volume
# ------------------------------
st.subheader("Banks Live vs UPI Volume")
fig2 = px.scatter(df, x="Banks_Live_Scaled", y="Volume_Mn",
                  labels={"Banks_Live_Scaled": "Banks Live (Scaled)", "Volume_Mn": "UPI Volume (Mn)"},
                  title="Banks Live vs Volume")
st.plotly_chart(fig2)

# ------------------------------
# Feature Importance
# ------------------------------
st.subheader("Feature Importance")
feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
st.bar_chart(feature_importance)
