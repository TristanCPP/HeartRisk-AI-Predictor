import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset from the local file
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


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Initialize the base Logistic Regression model
base_model = LogisticRegression(random_state=101, max_iter=1000)

# Train the model on the training data
base_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = base_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy * 100:.2f}%')
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)
