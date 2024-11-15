import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
import shap

# --- Load Dataset and Model ---
# Load the dataset
data = pd.read_csv('data/heart_disease_data.csv')

# Make a copy of the original dataset
data_copy = data.copy(deep=True)

# Dropping rows where Cholesterol is 0
data_copy = data_copy[(data_copy['Cholesterol'] != 0)]

# Load the pre-trained model and scaler
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

# Preprocess the dataset
X = data_copy.drop(columns=['HeartDisease'])
y = data_copy['HeartDisease']

# Define categorical data with all possible categories
X['ChestPainType'] = pd.Categorical(X['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
X['RestingECG'] = pd.Categorical(X['RestingECG'], categories=['Normal', 'ST', 'LVH'])
X['ST_Slope'] = pd.Categorical(X['ST_Slope'], categories=['Up', 'Flat', 'Down'])

# One-Hot Encode non-binary categorical variables without dropping any category
X = pd.get_dummies(X, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

temp_copy = X.copy(deep=True)

# Scale numerical features
scaler = pickle.load(open('scaler.pkl', 'rb'))
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)

print(X.info())
print(X.describe())
print(X.head())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)




# --- Dataset Visualization ---
# Uncomment desired visualization to use #

# # Correlation Matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(temp_copy.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title("Feature Correlation Matrix")
# plt.show()

# # Visualize feature distributions by Heart Disease
# features_to_plot = ['Age', 'Cholesterol', 'RestingBP', 'MaxHR']
# for feature in features_to_plot:
#     plt.figure(figsize=(8, 4))
#     sns.histplot(data=data_copy, x=feature, hue='HeartDisease', kde=True, bins=30, palette='viridis')
#     plt.title(f'{feature} Distribution by Heart Disease')
#     plt.show()

# # Count Plots for Categorical Features
# categorical_features = ['Sex', 'ChestPainType', 'ST_Slope', 'ExerciseAngina']
# for feature in categorical_features:
#     plt.figure(figsize=(6, 4))
#     sns.countplot(data=data_copy, x=feature, hue='HeartDisease', palette='viridis')
#     plt.title(f'{feature} Count by Heart Disease')
#     plt.show()

# # Count of Each Categorical Feature
# categorical_features = ['Sex', 'ChestPainType', 'ST_Slope', 'ExerciseAngina']
# for feature in categorical_features:
#     plt.figure(figsize=(6, 4))
#     sns.countplot(data=data_copy, x=feature, palette='viridis')
#     plt.title(f'Count of {feature}')
#     plt.ylabel('Count')
#     plt.xlabel(feature)
#     plt.xticks(rotation=45)
#     plt.show()

# # Distribution of Heart Disease
# plt.figure(figsize=(6, 4))
# sns.countplot(data=data_copy, x='HeartDisease', palette='viridis')
# plt.title('Distribution of Heart Disease (Target Variable)')
# plt.xlabel('HeartDisease (0: No, 1: Yes)')
# plt.ylabel('Count')
# plt.show()

# # Pairplot of Numerical Features Colored by HeartDisease
# numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
# sns.pairplot(data=data_copy, vars=numerical_features, hue='HeartDisease', palette='viridis')
# plt.suptitle('Pairplot of Numerical Features by HeartDisease', y=1.02)
# plt.show()

# # --- Model Visualization ---

# # Confusion Matrix
# ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()

# # Feature Importance
# importances = rf_model.feature_importances_
# feature_importance_df = pd.DataFrame({
#     'Feature': X_test.columns,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)
# plt.figure(figsize=(10, 6))
# sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
# plt.title('Feature Importance')
# plt.show()

# # ROC Curve
# RocCurveDisplay.from_estimator(rf_model, X_test, y_test)
# plt.title('ROC Curve')
# plt.show()
