import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('data/heart_disease_data.csv')

# Preprocessing
data_copy = data.copy(deep=True)

# Dropping rows where Cholesterol is 0
data_copy = data_copy[(data_copy['Cholesterol'] != 0)]

# Splitting features (X) and target variable (y)
X = data_copy.drop(columns=['HeartDisease'])
y = data_copy['HeartDisease']

# Define categorical data with all possible categories
X['ChestPainType'] = pd.Categorical(X['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
X['RestingECG'] = pd.Categorical(X['RestingECG'], categories=['Normal', 'ST', 'LVH'])
X['ST_Slope'] = pd.Categorical(X['ST_Slope'], categories=['Up', 'Flat', 'Down'])

# One-Hot Encode non-binary categorical variables without dropping any category
X = pd.get_dummies(X, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

# Scale numerical features
scaler = StandardScaler()
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Set up hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],                 # Regularization strength
    'solver': ['lbfgs', 'liblinear', 'saga'],     # Optimization solvers
}

# Grid Search
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
# print(f"Best Parameters: {best_params}")

# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Updated Logistic Regression Accuracy: {accuracy * 100:.2f}%')
print("Updated Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)