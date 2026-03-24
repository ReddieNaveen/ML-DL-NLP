import streamlit as st
import numpy as np
import pickle

# Load model
with open("model/model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

st.title("🏠 House Price Prediction App")

st.write("Enter the details below to predict house price")

# Inputs
income = st.number_input("Avg. Area Income", value=50000)
house_age = st.number_input("Avg. Area House Age", value=5)
rooms = st.number_input("Avg. Area Number of Rooms", value=6)
bedrooms = st.number_input("Avg. Area Number of Bedrooms", value=3)
population = st.number_input("Area Population", value=30000)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[income, house_age, rooms, bedrooms, population]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
