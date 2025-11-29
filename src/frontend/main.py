import streamlit as st
import requests
from typing import Dict, Any

# Configuración de la página
st.set_page_config(page_title="Stress Level Prediction", layout="centered")

# ========================
# Sidebar explicativo
# ========================
st.image("https://www.hipnosisvalencia.com/wp-content/uploads/2021/01/Tratamiento-de-la-ansiedad-por-hipnosis.jpg", use_container_width=True)
st.sidebar.title("About this App")
st.sidebar.info(
    """
    This application predicts the **stress level** of a person based on
    several lifestyle and health factors such as sleep quality, physical activity,
    coffee intake, BMI, smoking habits, alcohol consumption, heart rate, age, and occupation.
    
    Fill out the form on the main page and click **Predict** to get your predicted stress level.
    """
)

st.sidebar.markdown("---")

# ========================
# Título principal
# ========================
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Stress Level Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #306998;'>Predict your stress level based on lifestyle factors</h3>", unsafe_allow_html=True)
st.write("---")

# Opciones
COUNTRIES = [
    "Mexico", "USA", "Canada", "UK", "India", "Spain", "Germany",
    "France", "Italy", "Brazil", "China", "Japan", "Australia", "Norway", "Sweden",
    "Finland", "Belgium", "Netherlands", "Switzerland", "South Korea"
]

GENDERS = ["Female", "Male"]
SLEEP_QUALITY = ["Poor", "Fair", "Good", "Excellent"]
OCCUPATION = ["Office", "Other", "Student", "Healthcare", "Service"]
ALCOHOL_CONSUMPTION = ["Yes", "No"]
SMOKING = ["Yes", "No"]

# Función para construir payload
def build_sample(country, gender, age, coffee_intake, bmi, smoking, physical_activity_hours,
                 sleep_quality, alcohol_consumption, heart_rate, occupation) -> Dict[str, Any]:
    return {
        "country": country,
        "gender": gender,
        "age": age,
        "coffee_intake": coffee_intake,
        "bmi": bmi,
        "smoking": smoking,
        "physical_activity_hours": physical_activity_hours,
        "sleep_quality": sleep_quality,
        "alcohol_consumption": alcohol_consumption,
        "heart_rate": heart_rate,
        "occupation": occupation
    }

# ========================
# Formulario
# ========================
st.subheader("Fill the form to predict stress level")
with st.form("form_sample"):
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", COUNTRIES)
        gender = st.selectbox("Gender", GENDERS)
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        coffee_intake = st.number_input("Daily coffee intake (cups)", min_value=0, max_value=10, value=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1)

    with col2:
        smoking = st.selectbox("Smoking", SMOKING)
        physical_activity_hours = st.number_input("Physical activity hours", min_value=0.0, max_value=20.0, value=1.0)
        sleep_quality = st.selectbox("Sleep quality", SLEEP_QUALITY)
        alcohol_consumption = st.selectbox("Alcohol consumption", ALCOHOL_CONSUMPTION)
        heart_rate = st.number_input("Heart rate (bpm)", min_value=40, max_value=220, value=70)

    occupation = st.selectbox("Occupation", OCCUPATION)

    submitted = st.form_submit_button("Predict")

# ========================
# Predicción
# ========================
if submitted:
    input_dict = build_sample(
        country, gender, age, coffee_intake, bmi, smoking, physical_activity_hours,
        sleep_quality, alcohol_consumption, heart_rate, occupation
    )

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",  # En Docker usar host.docker.internal si API está en host
            json=input_dict,
            timeout=10
        )
        response.raise_for_status()
        prediction = response.json().get("prediction")
        st.success(f"### Predicted stress level: {prediction:.2f}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting API: {e}")



