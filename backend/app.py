from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the model, scaler, and feature names
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}

# Input schema
class UserInput(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

# Preprocess user data
def preprocess_user_data(user_data, feature_names, scaler):
    user_df = pd.DataFrame([user_data.dict()])

    # Preprocess categorical features
    user_df['ChestPainType'] = pd.Categorical(user_df['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
    user_df['RestingECG'] = pd.Categorical(user_df['RestingECG'], categories=['Normal', 'ST', 'LVH'])
    user_df['ST_Slope'] = pd.Categorical(user_df['ST_Slope'], categories=['Up', 'Flat', 'Down'])

    user_df = pd.get_dummies(user_df, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

    # Add missing columns
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df = user_df[feature_names]

    # Scale numerical features
    user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(
        user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
    )

    return user_df

# Map probability to risk category
def risk_category(probability):
    if probability < 0.2:
        return "Low Risk"
    elif 0.2 <= probability < 0.4:
        return "Slight Risk"
    elif 0.4 <= probability < 0.6:
        return "Moderate Risk"
    elif 0.6 <= probability < 0.8:
        return "High Risk"
    else:
        return "Extreme Risk"

# API route
@app.post("/predict")
def predict_risk(user_input: UserInput):
    user_df = preprocess_user_data(user_input, feature_names, scaler)
    probability = rf_model.predict_proba(user_df)[0][1]
    category = risk_category(probability)

    return {
        "risk_probability": probability,
        "risk_category": category
    }
