import joblib
import pandas as pd

# Load the saved model, scaler, and feature names
log_reg_model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Example user input data (replace with actual user inputs)
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

# Convert user data into DataFrame for preprocessing
user_df = pd.DataFrame([user_data])

# Preprocess categorical features with predefined categories
user_df['Sex'] = pd.Categorical(user_df['Sex'], categories=['F', 'M'])
user_df['ExerciseAngina'] = pd.Categorical(user_df['ExerciseAngina'], categories=['N', 'Y'])
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

# Predict risk probability
risk_probability = log_reg_model.predict_proba(user_df)[0][1]  # Probability of CHD
print(f"Predicted Coronary Heart Disease (CHD) Risk Probability: {risk_probability * 100:.2f}%")
