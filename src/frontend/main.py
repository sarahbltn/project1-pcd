import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any

st.set_page_config(page_title="Stress Level Prediction", layout="centered")

st.title("Stress Level Prediction")

st.write("""
### Application to predict stress level
""")

# -------------------------------
# OPCIONES UI
# -------------------------------
COUNTRIES = [
    "Mexico", "USA", "Canada", "UK", "India", "Spain", "Germany",
    "France", "Italy", "Brazil", "China", "Japan", "Australia", "Norway", "Sweden",
    "Finland", "Belgium", "Netherlands", "Switzerland", "South Korea"
]

GENDERS = ["Female", "Male"]
SLEEP_QUALITY = ["Poor", "Fair", "Good", "Excellent"]
OCCUPATION = ["Office", "Other", "Student", "Healthcare", "Service"]
ALCOHOL_CONSUMPTION = ["Yes", "No"]

CLASS_MAP = {
    "0": "Low Stress",
    "1": "Medium Stress",
    "2": "High Stress"
}

API_URL = "http://127.0.0.1:8000/api/v1/predict"


# -------------------------------
# FUNCIÓN PARA ARMAR PAYLOAD
# -------------------------------
def build_sample() -> Dict[str, Any]:

    alcohol_map = {
        "Yes": 1,
        "No": 0
    }

    return {
        "country": country,
        "gender": gender,
        "age": age,
        "coffee_intake": coffee_intake,
        "bmi": bmi,
        "smoking": smoking,
        "physical_activity_hours": physical_activity_hours,
        "sleep_quality": sleep_quality,
        "alcohol_consumption": alcohol_map[alcohol_consumption],
        "heart_rate": heart_rate,
        "occupation": occupation
    }


# -------------------------------
# FORMULARIO
# -------------------------------
st.subheader("Fill the form")

with st.form("form_sample"):
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", COUNTRIES)
        gender = st.selectbox("Gender", GENDERS)
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        coffee_intake = st.number_input("Daily Coffee Intake (cups)", min_value=0, max_value=10, value=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1)

    with col2:
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        physical_activity_hours = st.number_input("Physical Activity Hours", min_value=0.0, max_value=20.0, value=1.0)
        sleep_quality = st.selectbox("Sleep Quality", SLEEP_QUALITY)
        alcohol_consumption = st.selectbox("Alcohol Consumption", ALCOHOL_CONSUMPTION)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=220, value=70)

    occupation = st.selectbox("Occupation", OCCUPATION)

    btn = st.form_submit_button("Predict")



if btn:
    payload = build_sample()

    try:
        response = requests.post(API_URL)

        if response.status_code == 200:
            data = response.json()
            pred_class = data.get("prediction_class")
            pred_proba = data.get("prediction_proba")

            st.success(f"Prediction: **{CLASS_MAP.get(str(pred_class))}**")

            if pred_proba:
                dfp = pd.DataFrame(
                    {
                        "Class": ["Low", "Medium", "High"],
                        "Probability": pred_proba
                    }
                )
                st.table(dfp)

        else:
            st.error("Error: API responded with a non-200 status.")

    except:
        st.error("API not available.")
