import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Step 1: Load the dataset
data = pd.read_csv('data/heart_disease_data.csv')

# Step 2: Preprocess the data
processed_data = data[data['Cholesterol'] != 0]
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

# Step 5: Add polynomial and interaction features
poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=101)

# Step 7: Handle class imbalance with SMOTE
smote = SMOTE(random_state=101)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Step 8: Feature selection with RFE
log_reg = LogisticRegression(max_iter=500, solver='liblinear', C=0.1)  # Use Logistic Regression for RFE base estimator
rfe = RFE(log_reg, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train_sm, y_train_sm)
X_test_rfe = rfe.transform(X_test)

# Step 9: Hyperparameter tuning for XGBoost
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=101), param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1
)
xgb_search.fit(X_train_rfe, y_train_sm)

# Step 10: Evaluate the best XGBoost model
best_xgb = xgb_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_rfe)

print(f"Optimized XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
print(f"Best XGBoost Parameters: {xgb_search.best_params_}")
print(classification_report(y_test, y_pred_xgb))
