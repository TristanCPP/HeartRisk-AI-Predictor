import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the new dataset
data = pd.read_csv('data/heart_disease_data_test.csv')

# Make a copy of the original dataset
data_copy = data.copy(deep=True)

# Dropping rows where Cholesterol is 0
data_copy = data_copy[(data_copy['Cholesterol']!=0)]

X = data_copy.drop(columns=['HeartDisease'])
y = data_copy['HeartDisease']

# Define categorical data with all possible categories
X['ChestPainType'] = pd.Categorical(X['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
X['ST_Slope'] = pd.Categorical(X['ST_Slope'], categories=['Up', 'Flat', 'Down'])

# One-Hot Encode non-binary categorical variables without dropping any category
X = pd.get_dummies(X, columns=['Sex','ExerciseAngina', 'ChestPainType','ST_Slope'], dtype=int)

print(X.head())
# Scale numerical features
scaler = StandardScaler()
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Define parameter grid for KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1  # Use all available cores
)
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_knn = grid_search.best_estimator_
#print("Best KNN Parameters:", grid_search.best_params_)

# Evaluate the optimized KNN model
y_pred_knn = best_knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {accuracy * 100:.2f}%')
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

cm = confusion_matrix(y_test, y_pred_knn)
print(cm)
