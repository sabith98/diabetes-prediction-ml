import streamlit as st
import requests

st.write("Diabetes prediction")
# Form Elements
with st.form("Diabetes data"):
    pregnancies = st.text_input("No of pregnancies",placeholder="3")

    left, right = st.columns(2)
    glucose = left.text_input("Glucose level",placeholder="120")
    blood_pressure = right.text_input("Blood pressure",placeholder="70")
    skin_thickness = left.text_input("Skin thickness",placeholder="30")
    insulin = right.text_input("Insulin level",placeholder="40")
    bmi = st.text_input("BMI",placeholder="24.2")
    diabetes_pedigree_function = st.text_input("Diabetes pedigree function level",placeholder="0.361")
    age = st.text_input("Age",placeholder="24")

    # Submit button
    if st.form_submit_button("Send"):
        url="http://localhost:5000/predictdata"

        data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree_function,
            "Age": age
        }

        response = requests.post(url, json=data)

        st.write("Prediction: ", response.text)

