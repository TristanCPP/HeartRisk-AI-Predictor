# AI-Powered Heart Disease Risk Prediction – Backend

![Project Banner](https://img.shields.io/badge/Backend-HeartRiskAI-blue)

## Overview
This backend system powers a machine learning-based application that predicts a user's risk of developing Coronary Heart Disease (CHD). It uses a trained Random Forest model and exposes it through a Flask-based API. The backend receives XML-encoded user input, processes and preprocesses the data, and returns a risk score prediction and risk category. It integrates seamlessly with a Kotlin-based mobile frontend.

## Features
- **Random Forest Classifier**: Trained and optimized for high accuracy on curated CHD datasets.
- **REST API with XML Support**: Receives structured health input in XML, processes it, and responds with risk score output.
- **Model Serialization**: Trained model (`rf_model.pkl`), scaler (`scaler.pkl`), and feature structure (`feature_names.pkl`) saved for consistent deployment.
- **Preprocessing Utilities**: Ensures user inputs are validated, one-hot encoded, and scaled to match training data.
- **Test Interface**: Includes local simulation scripts for testing predictions independently of the frontend.

## Technologies Used
- **Python 3.12**
- **Flask** – Backend web framework and API handler
- **Scikit-learn** – Machine Learning model training and prediction
- **Pandas / NumPy** – Data preprocessing and manipulation
- **Joblib** – For saving and loading models
- **XML** – Used for structured API request/response formatting
- **unittest** – Python's built-in framework for unit and integration testing

## Project Structure
```
heart-risk-backend/
│
├── app.py                      # Flask API that handles XML prediction requests
├── rf_model.pkl                # Trained Random Forest model
├── scaler.pkl                  # Scaler used during model training
├── feature_names.pkl           # List of encoded/processed feature names
│
├── models/
│   └── random_forest_updated.py    # Main script for training and exporting the final model
│
├── data/
│   └── heart_disease_data_test.csv      # Cleaned dataset used for training/testing
│
├── visualizations/
│   ├── visualize_data_model.py     # Data visualization script
│   └── assets/                     # Output graphs and plots
│
├── tests/
│   ├── main_unitTesting.py
│   ├── random_forest_unitTesting.py
│   ├── main_integrationTesting.py
│   └── random_forest_integrationTesting.py
│
├── main_alt.py                # Local test script for simulating input and observing output
├── README.md
```

## API Usage

### Endpoint
```
POST /predict
Content-Type: application/xml
```

### Sample Request Body
```xml
<HeartRiskRequest>
    <Age>54</Age>
    <Sex>M</Sex>
    <ChestPainType>ASY</ChestPainType>
    <RestingBP>130</RestingBP>
    <Cholesterol>210</Cholesterol>
    <MaxHR>150</MaxHR>
    <ExerciseAngina>Y</ExerciseAngina>
    <Oldpeak>1.2</Oldpeak>
    <ST_Slope>Up</ST_Slope>
</HeartRiskRequest>
```

### Sample Response
```xml
<HeartRiskResponse>
    <RiskScore>64.29</RiskScore>
</HeartRiskResponse>
```

## Running the Server Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the API:
   ```bash
   python app.py
   ```

## Testing
Unit and integration tests are located in the `tests/` directory. To run a test:
```bash
python -m tests.main_unitTesting
```
You can modify and extend test files to simulate edge cases, verify scaling, or validate endpoint behavior.

## Final Notes
- Ensure all required files (`rf_model.pkl`, `scaler.pkl`, and `feature_names.pkl`) are present before starting the Flask app.
- This backend is designed to be paired with the mobile frontend, which handles input collection and UI presentation.
- The risk scoring model is based on medically-informed training data and provides feedback categories aligned with observed CHD risk tiers.

## Contributors
**Team Members**: Tristan Garner, Muhsen AbuMuhsen, Chase Lillard  
**Advisor**: Dr. [Professor's Name, optional]
