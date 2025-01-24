import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.joblib")

# Load the dataset to populate dropdowns
df = pd.read_csv("data.csv")

# Preprocess dropdown options
platforms = df["Platform"].unique().tolist()
countries = df["Country"].unique().tolist()
genders = ["Male", "Female", "Others"]

# Set the page configuration
st.set_page_config(layout="wide")

# Create input and result display
sideimage, gap1, leftmenu, gap2, rightmenu = st.columns([0.7, 0.1, 0.8, 0.1, 1.3])

# Display the side image
with sideimage:
        st.title("Social Media Misinformation")
        st.image("bg.jpg", width=100, use_container_width=True)

# Display a gap between the two columns
with gap1:
    st.write("")

# Input form
with leftmenu:
    st.header("Enter user details")
    platform = st.selectbox("Select Platform", platforms)
    country = st.selectbox("Select Country", countries)
    age = st.slider("Select Age", int(df["Age"].min()), int(df["Age"].max()))
    gender = st.selectbox("Select Gender", genders)

    prediction = None
    result = ""

    if st.button("Predict"):
        # Prepare dynamic one-hot encoding for input data
        input_data = pd.DataFrame({
            "Age": [age],
            **{f"Gender_{g}": 1 if g == gender else 0 for g in df["Gender"].unique()},
            **{f"Platform_{p}": 1 if p == platform else 0 for p in df["Platform"].unique()},
            **{f"Country_{c}": 1 if c == country else 0 for c in df["Country"].unique()}
        })

        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[model.feature_names_in_]

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Spreads misinformation" if prediction == 1 else "Does not spread misinformation"

# Display a gap between the two columns
with gap2:
    st.write("")

# Display the result in the right column
with rightmenu:
    st.header("Prediction result")
    if prediction is not None:  # Check if a prediction was made
        if prediction == 1:
            st.markdown(f"<span style='color: red; font-size:24px; font-weight: bold;'>{result}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color: green; font-size:24px; font-weight: bold;'>{result}</span>", unsafe_allow_html=True)
