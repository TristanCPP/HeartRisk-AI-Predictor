import joblib
import pandas as pd

# Load the saved model, scaler, and feature names
log_reg_model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

def get_user_input():
    user_data = {}
    
    user_data['Age'] = int(input("Enter Age: "))
    user_data['Sex'] = input("Enter Sex (M/F): ")
    user_data['ChestPainType'] = input("Enter Chest Pain Type (ATA/NAP/ASY/TA): ")
    user_data['RestingBP'] = int(input("Enter Resting Blood Pressure: "))
    user_data['Cholesterol'] = int(input("Enter Cholesterol: "))
    user_data['FastingBS'] = int(input("Enter Fasting Blood Sugar (0/1): "))
    user_data['RestingECG'] = input("Enter Resting ECG (Normal/ST/LVH): ")
    user_data['MaxHR'] = int(input("Enter Max Heart Rate: "))
    user_data['ExerciseAngina'] = input("Enter Exercise Angina (Y/N): ")
    user_data['Oldpeak'] = float(input("Enter Oldpeak: "))
    user_data['ST_Slope'] = input("Enter ST Slope (Up/Flat/Down): ")
    
    return user_data

# Get user input
user_data = get_user_input()

# Convert user data into DataFrame for preprocessing
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

# Predict risk probability
risk_probability = log_reg_model.predict_proba(user_df)[0][1]  # Probability of CHD
print(f"Predicted Coronary Heart Disease (CHD) Risk Probability: {risk_probability * 100:.2f}%")