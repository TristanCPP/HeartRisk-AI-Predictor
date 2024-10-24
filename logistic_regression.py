import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset from the local file
data = pd.read_csv('Heart_disease_cleveland_new.csv')

# Replace '?' with NaN and drop rows with missing values
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Convert categorical columns to numerical (e.g., sex, cp, etc.)
data['sex'] = data['sex'].astype(int)
data['cp'] = data['cp'].astype(int)
data['fbs'] = data['fbs'].astype(int)
data['restecg'] = data['restecg'].astype(int)
data['exang'] = data['exang'].astype(int)
data['slope'] = data['slope'].astype(int)
data['ca'] = data['ca'].astype(int)
data['thal'] = data['thal'].astype(int)

# We remove 'oldpeak' and 'ca' for this experiment based on their correlations
selected_features = ['age', 'sex', 'cp', 'thalach', 'exang', 'slope', 'thal']

# Split features (X) and target (y)
X = data[selected_features]
y = data['target']

# Standardize features (scale numerical data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Get the predicted probabilities for each sample in the test set
#   y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class (heart disease)

# Define risk tiers based on probability thresholds
# def categorize_risk(prob):
#     if prob < 0.2:
#         return 'Low Risk (Green)'
#     elif prob < 0.4:
#         return 'Slight Risk (Yellow)'
#     elif prob < 0.6:
#         return 'Moderate Risk (Orange)'
#     elif prob < 0.8:
#         return 'High Risk (Dark Orange)'
#     else:
#         return 'Extreme Risk (Bright Red)'
    
# Apply the categorization to the predicted probabilities
#   risk_categories = [categorize_risk(prob) for prob in y_probs]

# Print the first few results
#   print(risk_categories[:10])

# Print the probabilities for the the test data
#   print(y_probs)

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

print("DAWGS")