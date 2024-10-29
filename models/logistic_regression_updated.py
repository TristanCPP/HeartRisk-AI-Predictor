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

# Step 3: Label Encoding for binary categorical variables
encoder = LabelEncoder()

X['Sex'] = encoder.fit_transform(X['Sex'])
X['ChestPainType'] = encoder.fit_transform(X['ChestPainType'])
X['RestingECG'] = encoder.fit_transform(X['RestingECG'])
X['ExerciseAngina'] = encoder.fit_transform(X['ExerciseAngina'])
X['ST_Slope'] = encoder.fit_transform(X['ST_Slope'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train , X_test , y_train , y_test = train_test_split(X_scaled,y ,test_size=0.2 , random_state=101)

log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

y_pred = log_reg_model.predict(X_test)

print(accuracy_score(y_test, y_pred))