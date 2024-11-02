#USES CLEVELAND DATASET

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset from the local file
data = pd.read_csv('data/Heart_disease_cleveland_new.csv')

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
# X = data[selected_features]
y = data['target']

# Standardize features (scale numerical data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# **Set up the GridSearchCV** to tune hyperparameters
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['lbfgs', 'liblinear']  # Solvers for optimization
}

# Initialize GridSearchCV with LogisticRegression
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# **Print the best parameters and the best cross-validation score**
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

# **Train the model with the best parameters**
best_model = grid_search.best_estimator_

# Make predictions with the optimized model
y_pred_best = best_model.predict(X_test)

# Evaluate the model’s accuracy on the test set
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Test Set Accuracy with Best Parameters: {accuracy_best * 100:.2f}%")

# Train the model
#   model.fit(X_train, y_train)

# Make predictions on the test set
#   y_pred = model.predict(X_test)

# Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')

