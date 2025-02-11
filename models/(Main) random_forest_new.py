import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# Initialize the RandomForestClassifier with optimized parameters for better accuracy
rf_model = RandomForestClassifier(
    n_estimators=250,          
    criterion='entropy', 
    max_depth=10,              # Limit the depth of each tree to prevent overfitting
    min_samples_split=10,      # Require a minimum of 10 samples to split a node
    min_samples_leaf=4,        # Require at least 4 samples at each leaf node
    max_features='sqrt',       # Consider a subset of features (log2) for each split
    random_state=101,
    bootstrap=True,            # to reduce variance
)

# Train the model on the entire training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Updated Random Forest Accuracy: {accuracy * 100:.2f}%')

# Print a detailed classification report
print("Updated Random Forest Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# # Exporting Model
# with open('rf_model.pkl', 'wb') as model_file:
#     pickle.dump(rf_model, model_file)
# with open('scaler.pkl', 'wb') as scaler_file:
#     pickle.dump(scaler, scaler_file)
# with open('feature_names.pkl', 'wb') as feature_file:
#     pickle.dump(X.columns.tolist(), feature_file)
