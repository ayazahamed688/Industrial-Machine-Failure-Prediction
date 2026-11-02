
import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature names
model = joblib.load("models/failure_model.pkl")
features = joblib.load("models/features.pkl")

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="🏭",
    layout="centered"
)

st.title("🏭 Industrial Machine Failure Prediction")
st.write("Predict whether an industrial machine is likely to fail.")

st.subheader("Enter Machine Details")

machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

air_temp = st.number_input(
    "Air Temperature (K)", min_value=250.0, max_value=350.0, value=300.0
)

process_temp = st.number_input(
    "Process Temperature (K)", min_value=250.0, max_value=400.0, value=310.0
)

rotational_speed = st.number_input(
    "Rotational Speed (rpm)", min_value=0, max_value=5000, value=1500
)

torque = st.number_input(
    "Torque (Nm)", min_value=0.0, max_value=200.0, value=40.0
)

tool_wear = st.number_input(
    "Tool Wear (minutes)", min_value=0, max_value=300, value=100
)

if st.button("Predict Machine Failure"):
    input_data = pd.DataFrame(
        [[air_temp, process_temp, rotational_speed, torque,
          tool_wear, int(machine_type == "L"), int(machine_type == "M")]],
        columns=[
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type_L",
            "Type_M"
        ]
    )

    input_data = input_data.reindex(columns=features, fill_value=0)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Machine failure predicted!")
    else:
        st.success("✅ No machine failure predicted.")
