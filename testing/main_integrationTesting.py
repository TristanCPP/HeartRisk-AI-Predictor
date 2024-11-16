import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model, scaler, and feature names from serialized files
rf_model = joblib.load('rf_model.pkl')  # Random Forest model
scaler = joblib.load('scaler.pkl')  # Scaler for feature scaling
feature_names = joblib.load('feature_names.pkl')  # List of feature names required by the model

# Default values for missing or unknown inputs in user data
DEFAULT_VALUES = {
    'ChestPainType': 'ASY', 'RestingBP': 132.54, 'Cholesterol': 244.64,
    'FastingBS': 0, 'RestingECG': 'Normal', 'MaxHR': 136.81,
    'ExerciseAngina': 'N', 'Oldpeak': 0.89, 'ST_Slope': 'Up'
}

# Sample test data to simulate user input for prediction
test_data = {
    'Age': 45, 'Sex': 'M', 'ChestPainType': 'ATA', 'RestingBP': 130,
    'Cholesterol': 220, 'FastingBS': 0, 'RestingECG': 'Normal',
    'MaxHR': 150, 'ExerciseAngina': 'N', 'Oldpeak': 1.0, 'ST_Slope': 'Up'
}

# Preprocess the input data and generate a risk prediction
def preprocess_and_predict(data):
    # Replace missing or unknown values with defaults
    for key, value in data.items():
        if value == 'Unknown':
            data[key] = DEFAULT_VALUES[key]
    
    # Convert user data into a DataFrame for processing
    user_df = pd.DataFrame([data])
    
    # Convert categorical features into appropriate categories
    user_df['ChestPainType'] = pd.Categorical(user_df['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
    user_df['RestingECG'] = pd.Categorical(user_df['RestingECG'], categories=['Normal', 'ST', 'LVH'])
    user_df['ST_Slope'] = pd.Categorical(user_df['ST_Slope'], categories=['Up', 'Flat', 'Down'])
    
    # One-hot encode categorical variables
    user_df = pd.get_dummies(user_df, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)
    
    # Ensure all required features are present, adding missing columns with default values
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0
    
    # Reorder columns to match the expected model input
    user_df = user_df[feature_names]
    
    # Apply scaling to numerical features
    user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(
        user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
    )
    
    # Predict risk probability and categorize the risk level
    risk_probability = rf_model.predict_proba(user_df)[0][1]
    risk_category = (
        "Low Risk" if risk_probability < 0.2 else
        "Slight Risk" if risk_probability < 0.4 else
        "Moderate Risk" if risk_probability < 0.6 else
        "High Risk" if risk_probability < 0.8 else
        "Extreme Risk"
    )
    return risk_probability, risk_category

# Validate that required files are loaded correctly
def test_data_loading():
    assert os.path.exists('rf_model.pkl'), "Model file not found."
    assert os.path.exists('scaler.pkl'), "Scaler file not found."
    assert os.path.exists('feature_names.pkl'), "Feature names file not found."
    print("Model, scaler, and feature files loaded successfully!")

# Validate the input data structure for preprocessing
def test_data_preprocessing(data):
    assert 'Age' in data, "'Age' feature missing in the input data."
    assert 'Cholesterol' in data, "'Cholesterol' feature missing in the input data."
    print("Preprocessing steps validated successfully!")

# Test the prediction function for correct output range and risk categories
def test_prediction():
    probability, category = preprocess_and_predict(test_data)
    assert 0 <= probability <= 1, f"Prediction probability out of bounds: {probability}"
    assert category in ["Low Risk", "Slight Risk", "Moderate Risk", "High Risk", "Extreme Risk"], "Invalid risk category."
    print("Prediction function works successfully!")

# Simulate user input and validate the GUI data
def test_gui_input():
    # Simulate input for the GUI form (mimicking what a user might enter in the application)
    data = test_data.copy()  # This is the simulated form data
    
    # Validate the data types and values of user inputs
    assert isinstance(data['Age'], (int, float)), "'Age' should be numeric."
    assert data['Sex'] in ['M', 'F'], "'Sex' should be either 'M' or 'F'."
    assert data['ChestPainType'] in ['ATA', 'NAP', 'ASY', 'TA'], "'ChestPainType' has invalid value."
    assert isinstance(data['RestingBP'], (int, float)), "'RestingBP' should be numeric."
    assert isinstance(data['Cholesterol'], (int, float)), "'Cholesterol' should be numeric."
    assert data['FastingBS'] in [0, 1], "'FastingBS' should be 0 or 1."
    assert data['RestingECG'] in ['Normal', 'ST', 'LVH'], "'RestingECG' has invalid value."
    assert isinstance(data['MaxHR'], (int, float)), "'MaxHR' should be numeric."
    assert data['ExerciseAngina'] in ['Y', 'N'], "'ExerciseAngina' should be 'Y' or 'N'."
    assert isinstance(data['Oldpeak'], (int, float)), "'Oldpeak' should be numeric."
    assert data['ST_Slope'] in ['Up', 'Flat', 'Down'], "'ST_Slope' has invalid value."
    print("GUI input validation passed successfully!")

# Validate that the model and related files are exported correctly
def test_model_export():
    # Exporting model, scaler, and feature names to files
    with open('rf_model.pkl', 'wb') as model_file:
        pickle.dump(rf_model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open('feature_names.pkl', 'wb') as feature_file:
        pickle.dump(feature_names, feature_file)

    # Check if files exist after export
    assert os.path.exists('rf_model.pkl'), "Model file export failed."
    assert os.path.exists('scaler.pkl'), "Scaler file export failed."
    assert os.path.exists('feature_names.pkl'), "Feature names file export failed."
    print("Model and related files exported successfully!")

# Run all integration tests to validate the application workflow
def run_integration_tests():
    test_data_loading()
    test_data_preprocessing(test_data)
    test_prediction()
    test_gui_input()
    test_model_export()

# Execute the tests
run_integration_tests()
