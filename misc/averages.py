# Used for calculating the averages used in unknown or empty fields when user enters data

import pandas as pd

# Load your dataset
data = pd.read_csv('data/heart_disease_data.csv')

# Calculate the most common (mode) for categorical features
most_common_values = {
    'Sex': data['Sex'].mode()[0],
    'ChestPainType': data['ChestPainType'].mode()[0],
    'RestingECG': data['RestingECG'].mode()[0],
    'ExerciseAngina': data['ExerciseAngina'].mode()[0],
    'ST_Slope': data['ST_Slope'].mode()[0]
}

# Calculate the average (mean) for numerical features
average_values = {
    'Age': data['Age'].mean(),
    'RestingBP': data['RestingBP'][data['RestingBP'] > 0].mean(),  # Exclude 0 values if they represent missing data
    'Cholesterol': data['Cholesterol'][data['Cholesterol'] > 0].mean(),  # Exclude 0 values if they represent missing data
    'MaxHR': data['MaxHR'].mean(),
    'Oldpeak': data['Oldpeak'].mean()
}

# Combine the results
default_values = {**most_common_values, **average_values}

# Print the default values
for feature, value in default_values.items():
    print(f"{feature}: {value:.2f}" if isinstance(value, float) else f"{feature}: {value}")
