
# Diamond Price Prediction using KNN and Streamlit

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

# Load the trained pipeline
MODEL_PATH = Path(__file__).parent / "models" / "knn_diamond_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

model = load_model()

# Application heading
st.title("💎 Diamond Price Prediction")
st.write(
    "Predict the estimated price of a diamond using "
    "a trained K-Nearest Neighbors (KNN) regression model."
)

st.divider()

# Collect diamond characteristics
st.subheader("Enter Diamond Details")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input(
        "Carat", min_value=0.1, max_value=5.0,
        value=0.5, step=0.1
    )

    cut = st.selectbox(
        "Cut",
        ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    )

    color = st.selectbox(
        "Color", ["D", "E", "F", "G", "H", "I", "J"]
    )

    clarity = st.selectbox(
        "Clarity",
        ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    )

with col2:
    depth = st.number_input(
        "Depth (%)", min_value=40.0, max_value=80.0,
        value=61.5, step=0.1
    )

    table = st.number_input(
        "Table (%)", min_value=40.0, max_value=100.0,
        value=57.0, step=0.1
    )

    x = st.number_input(
        "Length (mm)", min_value=0.0, max_value=15.0,
        value=5.0, step=0.1
    )

    y = st.number_input(
        "Width (mm)", min_value=0.0, max_value=15.0,
        value=5.0, step=0.1
    )

    z = st.number_input(
        "Depth (mm)", min_value=0.0, max_value=10.0,
        value=3.1, step=0.1
    )

# Generate prediction
if st.button("Predict Diamond Price", type="primary"):

    input_data = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.metric(
            "Estimated Diamond Price",
            f"${prediction:,.2f}"
        )

        st.caption(
            "This is a machine-learning estimate, not a guaranteed market price."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")








        
