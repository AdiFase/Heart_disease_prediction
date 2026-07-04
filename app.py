import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_artifacts():
    try:
        model            = joblib.load("logistic_heart_disease_prediction_model.pkl")
        scaler           = joblib.load("scaler.pkl")
        expected_columns = joblib.load("cols.pkl")
        return model, scaler, expected_columns
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        st.stop()

model, scaler, expected_columns = load_artifacts()

st.title("Heart Disease Prediction")

# ── Inputs ────────────────────────────────────────────────────────────────────
age             = st.number_input("Age", 1, 120, 40)
sex             = st.selectbox("Sex", ["Female", "Male"])
resting_bp      = st.number_input("Resting Blood Pressure", 0, 250, 120)
cholesterol     = st.number_input("Cholesterol (mm/dl)", 0, 700, 200)
maxhr           = st.number_input("MaxHR", 50, 250, 150)
oldpeak         = st.number_input("Oldpeak", -3.0, 10.0, 1.0)
exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
chest_pain      = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_ecg     = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
st_slope        = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    # STEP 1: Compute Age_Oldpeak from RAW values (before scaling)
    age_oldpeak = age * oldpeak

    # STEP 2: Build scaler input — exactly 6 columns in exact order scaler was fitted on
    raw_df = pd.DataFrame([{
        "Age":         age,
        "RestingBP":   resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR":       maxhr,
        "Oldpeak":     oldpeak,
        "Age_Oldpeak": age_oldpeak,
    }])

    scaled_values = scaler.transform(raw_df)
    scaled_df     = pd.DataFrame(scaled_values, columns=raw_df.columns)

    # STEP 3: Build the 17-column model input
    data = dict.fromkeys(expected_columns, 0)

    # Scaled numerics that exist in expected_columns
    data["Age"]         = scaled_df["Age"].iloc[0]
    data["MaxHR"]       = scaled_df["MaxHR"].iloc[0]
    data["Oldpeak"]     = scaled_df["Oldpeak"].iloc[0]
    data["Age_Oldpeak"] = scaled_df["Age_Oldpeak"].iloc[0]

    # Binary features
    data["Sex"]             = 1 if sex == "Male" else 0
    data["ExerciseAngina"]  = 1 if exercise_angina == "Yes" else 0
    data["FastingBS"]       = 1 if fasting_bs == "Yes" else 0
    data["High_BP"]         = 1 if oldpeak > 2 else 0

    # Age group
    if age < 40:
        data["Age_Group_Young"]       = 1
    elif age < 60:
        data["Age_Group_Middle_Aged"] = 1
    else:
        data["Age_Group_Senior"]      = 1

    # One-hot: Chest Pain
    cp_col = f"ChestPainType_{chest_pain}"
    if cp_col in data:
        data[cp_col] = 1

    # One-hot: Resting ECG
    ecg_col = f"RestingECG_{resting_ecg}"
    if ecg_col in data:
        data[ecg_col] = 1

    # One-hot: ST Slope
    slope_col = f"ST_Slope_{st_slope}"
    if slope_col in data:
        data[slope_col] = 1

    # STEP 4: Enforce exact column order the model expects
    input_df = pd.DataFrame([data])[expected_columns]

    try:
        prediction  = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        if prediction[0] == 1:
            st.error(f"❤️ Heart Disease Detected  (confidence: {probability:.1%})")
        else:
            st.success(f"✅ No Heart Disease Detected  (confidence: {1 - probability:.1%})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Model expects:", list(expected_columns))
        st.write("Got:", list(input_df.columns))


        