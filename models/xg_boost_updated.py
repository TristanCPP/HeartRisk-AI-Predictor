import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
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

# One-Hot Encode non-binary categorical variables without dropping any category
X = pd.get_dummies(X, columns=['Sex','ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

# Scale numerical features
scaler = StandardScaler()
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)

print(X.head())

# Step 5: Add polynomial and interaction features
# poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
# X_poly = poly.fit_transform(X)

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# # Step 7: Handle class imbalance with SMOTE
# smote = SMOTE(random_state=101)
# X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# # Step 8: Feature selection with SelectKBest
# selector = SelectKBest(mutual_info_classif, k=10)
# X_train_best = selector.fit_transform(X_train, y_train)
# X_test_best = selector.transform(X_test)

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
xgb_search.fit(X_train, y_train)

# Step 10: Evaluate the best XGBoost model
best_xgb = xgb_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

print(f"Optimized XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
print(f"Best XGBoost Parameters: {xgb_search.best_params_}")
print(classification_report(y_test, y_pred_xgb))
