import joblib
import pandas as pd

# Function: Preprocess user data
def preprocess_user_data(user_data, feature_names, scaler):
    """
    Preprocess user input to match the training data structure.

    Args:
        user_data (dict): Dictionary containing user input.
        feature_names (list): List of features used in the model.
        scaler (StandardScaler): Scaler used to standardize numerical features.

    Returns:
        pd.DataFrame: Preprocessed user input.
    """
    # Convert user data into DataFrame
    user_df = pd.DataFrame([user_data])

    # Preprocess categorical features with predefined categories
    user_df['ChestPainType'] = pd.Categorical(user_df['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
    user_df['RestingECG'] = pd.Categorical(user_df['RestingECG'], categories=['Normal', 'ST', 'LVH'])
    user_df['ST_Slope'] = pd.Categorical(user_df['ST_Slope'], categories=['Up', 'Flat', 'Down'])

    # Apply one-hot encoding to match the training data
    user_df = pd.get_dummies(user_df, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

    # Add any missing columns from the training features, setting them to 0
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0

    # Reorder columns to match training data order
    user_df = user_df[feature_names]

    # Scale numerical features
    user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(
        user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
    )

    return user_df


# Function: Determine risk category
def risk_category(probability):
    """
    Map risk probability to risk category.

    Args:
        probability (float): Predicted risk probability.

    Returns:
        str: Risk category.
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


# Load the saved model, scaler, and feature names
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Example user input data (replace with actual user input)
user_data = {
    'Age': 55,
    'Sex': 'M',
    'ChestPainType': 'ATA',
    'RestingBP': 140,
    'Cholesterol': 220,
    'FastingBS': 1,
    'RestingECG': 'Normal',
    'MaxHR': 150,
    'ExerciseAngina': 'Y',
    'Oldpeak': 1.5,
    'ST_Slope': 'Up'
}

# Preprocess user data
user_df = preprocess_user_data(user_data, feature_names, scaler)

# Predict risk probability
risk_probability = rf_model.predict_proba(user_df)[0][1]  # Probability of CHD
category = risk_category(risk_probability)

# Output results
print(f"Predicted Coronary Heart Disease (CHD) Risk Probability: {risk_probability * 100:.2f}%")
print(f"Risk Category: {category}")
