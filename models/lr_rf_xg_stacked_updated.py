import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Step 1: Load the dataset
data = pd.read_csv('data/heart_disease_data.csv')

# Step 2: Preprocess the data

processed_data = data.copy(deep=True)
processed_data = processed_data[processed_data['Cholesterol'] != 0]

X = processed_data.drop(columns=['HeartDisease'])
y = processed_data['HeartDisease']

# Define categorical data with all possible categories
X['ChestPainType'] = pd.Categorical(X['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
X['RestingECG'] = pd.Categorical(X['RestingECG'], categories=['Normal', 'ST', 'LVH'])
X['ST_Slope'] = pd.Categorical(X['ST_Slope'], categories=['Up', 'Flat', 'Down'])

# One-Hot Encode non-binary categorical variables without dropping any category
X = pd.get_dummies(X, columns=['Sex','ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

# Step 4: Scale numerical features
scaler = StandardScaler()
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)

print(X.head())

# Step 5: Add polynomial and interaction features
poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=101)

# Step 7: Handle class imbalance with SMOTE
smote = SMOTE(random_state=101)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Step 8: Feature selection with SelectKBest
selector = SelectKBest(mutual_info_classif, k=10)  # Select top 10 features
X_train_best = selector.fit_transform(X_train_sm, y_train_sm)
X_test_best = selector.transform(X_test)

# Step 9: Define the base models
log_reg = LogisticRegression(max_iter=500, solver='liblinear', C=0.1)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=101)

# XGBoost with hyperparameter tuning
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
xgb_search.fit(X_train_best, y_train_sm)
best_xgb = xgb_search.best_estimator_

# Step 10: Define the Stacking Classifier
estimators = [
    ('log_reg', log_reg),
    ('random_forest', rf_model),
    ('xgboost', best_xgb)
]

stacking_clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(max_iter=1000)
)

# Step 11: Train and evaluate the Stacking Classifier
stacking_clf.fit(X_train_best, y_train_sm)
y_pred_stack = stacking_clf.predict(X_test_best)

print(f"Stacking Classifier Accuracy: {accuracy_score(y_test, y_pred_stack) * 100:.2f}%")
print("Stacking Classifier Classification Report:\n", classification_report(y_test, y_pred_stack))
