import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Cleveland dataset
data = pd.read_csv('Heart_disease_cleveland_new.csv')

# Handle missing values
data = data.dropna()

# Convert categorical columns to numerical
data['sex'] = data['sex'].astype(int)
data['cp'] = data['cp'].astype(int)
data['fbs'] = data['fbs'].astype(int)
data['restecg'] = data['restecg'].astype(int)
data['exang'] = data['exang'].astype(int)
data['slope'] = data['slope'].astype(int)
data['ca'] = data['ca'].astype(int)
data['thal'] = data['thal'].astype(int)

# Define features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree with improved hyperparameters
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

# Cross-validation to evaluate the model
cv_scores = cross_val_score(dt_model, X_scaled, y, cv=5)
print(f'Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%')

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Get the predicted probabilities for each sample in the test set
y_probs = dt_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (heart disease)

# Define risk tiers based on probability thresholds
def categorize_risk(prob):
    if prob < 0.2:
        return 'Low Risk (Green)'
    elif prob < 0.4:
        return 'Slight Risk (Yellow)'
    elif prob < 0.6:
        return 'Moderate Risk (Orange)'
    elif prob < 0.8:
        return 'High Risk (Dark Orange)'
    else:
        return 'Extreme Risk (Bright Red)'

# Apply the categorization to the predicted probabilities
risk_categories = [categorize_risk(prob) for prob in y_probs]

# Print the first few risk categories
print(risk_categories[:10])

# Print the predicted probabilities for the test data
print(y_probs)
