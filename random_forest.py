import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Split features (X) and target (y)
X = data.drop(columns='target')
y = data['target']

# Standardize features (scale numerical data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier with additional parameters
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',  # You could also try 'entropy'
    max_depth=None,  # Control the maximum depth of the trees
    min_samples_split=2,  # Minimum samples to split an internal node
    min_samples_leaf=1,  # Minimum samples at a leaf node
    max_features='sqrt',  # Number of features to consider for the best split
    random_state=42  # For reproducibility
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print a detailed classification report
print(classification_report(y_test, y_pred))

# Get the predicted probabilities for each sample in the test set
y_probs = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (heart disease)

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

# Print the first few results
print(risk_categories[:10])

# Print the probabilities for the test data
print(y_probs)

# Plot the feature correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()
