import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Step 1: Load the dataset
data = pd.read_csv('data/heart_disease_data.csv')

# Step 2: Preprocess the data
processed_data = data[data['Cholesterol'] != 0]
X = processed_data.drop(columns=['HeartDisease'])
y = processed_data['HeartDisease']

# Define categorical data with all possible categories
X['ChestPainType'] = pd.Categorical(X['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
X['RestingECG'] = pd.Categorical(X['RestingECG'], categories=['Normal', 'ST', 'LVH'])
X['ST_Slope'] = pd.Categorical(X['ST_Slope'], categories=['Up', 'Flat', 'Down'])

# Step 3: One-Hot Encode categorical features
X = pd.get_dummies(X, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

# Step 4: Scale numerical features
scaler = StandardScaler()
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)

print(X.head())

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Step 6: Define the base models
log_reg = LogisticRegression(max_iter=500, solver='liblinear')
rf_model = RandomForestClassifier(n_estimators=100, random_state=101)
xgb_model = XGBClassifier(random_state=101)

# Step 7: Define the Stacking Classifier
estimators = [
    ('log_reg', log_reg),
    ('random_forest', rf_model),
    ('xgboost', xgb_model)
]

stacking_clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(max_iter=1000)
)

# Step 8: Train and evaluate the Stacking Classifier
stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)

print(f"Stacking Classifier Accuracy: {accuracy_score(y_test, y_pred_stack)}")
print(classification_report(y_test, y_pred_stack))
