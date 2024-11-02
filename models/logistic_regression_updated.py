import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv('data/heart_disease_data.csv')

# Step 2: Make a copy of the original dataset
processed_data = data.copy(deep=True)
# print(processed_data.head())
# print(processed_data.info())
# print(processed_data.describe())

processed_data = processed_data[(processed_data['Cholesterol']!=0)]

X = processed_data.drop(columns=['HeartDisease'])
y = processed_data['HeartDisease']

# Step 3: Encode binary and categorical features
X['Sex'] = X['Sex'].map({'M': 1, 'F': 0})
X['ExerciseAngina'] = X['ExerciseAngina'].map({'Y': 1, 'N': 0})
X = pd.get_dummies(X, columns=['ChestPainType', 'ST_Slope', 'RestingECG'], drop_first=True)

# Step 4: Scale numerical features
scaler = StandardScaler()
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)


X_train , X_test , y_train , y_test = train_test_split(X ,y ,test_size=0.2 , random_state=101)

log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train) 

y_pred = log_reg_model.predict(X_test)

print(accuracy_score(y_test, y_pred))