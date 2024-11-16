from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the machine learning model, scaler, and feature names from serialized files
rf_model = joblib.load('rf_model.pkl')  # Random Forest model for heart disease prediction
scaler = joblib.load('scaler.pkl')  # Scaler to normalize numerical features
feature_names = joblib.load('feature_names.pkl')  # List of feature names expected by the model

# Initialize a FastAPI application
app = FastAPI()

# Root endpoint to check API status
@app.get("/")
def read_root():
    """
    Root endpoint to confirm the API is running.
    Returns a simple message indicating the API status.
    """
    return {"message": "API is running"}

# Define the schema for user input using Pydantic
class UserInput(BaseModel):
    """
    Schema for validating user input.
    Includes all required fields for the prediction model.
    """
    Age: int  # User's age
    Sex: str  # User's sex (e.g., 'M' or 'F')
    ChestPainType: str  # Type of chest pain (e.g., 'ATA', 'NAP', etc.)
    RestingBP: int  # Resting blood pressure
    Cholesterol: int  # Cholesterol level
    FastingBS: int  # Fasting blood sugar (0 or 1)
    RestingECG: str  # Resting ECG result (e.g., 'Normal', 'ST', etc.)
    MaxHR: int  # Maximum heart rate achieved
    ExerciseAngina: str  # Exercise-induced angina ('Y' or 'N')
    Oldpeak: float  # ST depression induced by exercise
    ST_Slope: str  # Slope of the peak exercise ST segment (e.g., 'Up', 'Flat', etc.)

# Function to preprocess user data for prediction
def preprocess_user_data(user_data, feature_names, scaler):
    """
    Preprocess the user input data to prepare it for prediction.
    Includes one-hot encoding, handling missing columns, and scaling numerical features.

    Args:
        user_data: User input data as a dictionary.
        feature_names: List of features expected by the model.
        scaler: Scaler object for normalizing numerical features.

    Returns:
        Preprocessed data as a Pandas DataFrame.
    """
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_data.dict()])

    # Convert categorical features into appropriate categories
    user_df['ChestPainType'] = pd.Categorical(user_df['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
    user_df['RestingECG'] = pd.Categorical(user_df['RestingECG'], categories=['Normal', 'ST', 'LVH'])
    user_df['ST_Slope'] = pd.Categorical(user_df['ST_Slope'], categories=['Up', 'Flat', 'Down'])

    # Apply one-hot encoding to categorical variables
    user_df = pd.get_dummies(user_df, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

    # Add any missing columns that the model expects but are not present in the user data
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0

    # Reorder the columns to match the model's expectations
    user_df = user_df[feature_names]

    # Scale numerical features for consistency with model training
    user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(
        user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
    )

    return user_df

# Function to map risk probability to a risk category
def risk_category(probability):
    """
    Map the risk probability to a user-friendly risk category.

    Args:
        probability: Risk probability value (0 to 1).

    Returns:
        A string representing the risk category.
    """
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

# API endpoint to predict heart disease risk
@app.post("/predict")
def predict_risk(user_input: UserInput):
    """
    Predict the heart disease risk based on user input.

    Args:
        user_input: Input data from the user, validated against the UserInput schema.

    Returns:
        A JSON object with the predicted risk probability and risk category.
    """
    # Preprocess user input data
    user_df = preprocess_user_data(user_input, feature_names, scaler)
    
    # Predict the risk probability using the trained model
    probability = rf_model.predict_proba(user_df)[0][1]
    
    # Map the probability to a risk category
    category = risk_category(probability)

    # Return the risk probability and category as a response
    return {
        "risk_probability": probability,
        "risk_category": category
    }
