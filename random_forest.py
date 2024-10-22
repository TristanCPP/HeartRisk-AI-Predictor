˜˜import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Cleveland dataset (replace 'cleveland.csv' with your actual file path)
data = pd.read_csv("Heart_disease_cleveland_new.csv")

# Display the first few rows of the dataset
print(data)

# Check for missing values and handle them
data = data.dropna()

# Define features (X) and target (y) variables
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print a detailed classification report
print(classification_report(y_test, y_pred))

# Define the feature names from the training data
feature_names = X_train.columns

# New data input considered as the user input
new_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
new_data_df = pd.DataFrame(new_data, columns=feature_names)

# Make predictions on the new data
new_prediction = rf_model.predict(new_data_df)

# Output the result
if new_prediction[0] == 1:
    print("High risk of heart disease")
else:
    print("Low risk of heart disease")
