import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset and create a copy for preprocessing
data = pd.read_csv('data/heart_disease_data.csv')  # Load heart disease dataset
data_copy = data.copy(deep=True)  # Create a deep copy of the dataset for safe preprocessing

# Preprocessing: Drop rows with invalid cholesterol values
data_copy = data_copy[(data_copy['Cholesterol'] != 0)]  # Remove rows with zero cholesterol values

# Split features (X) and target variable (y)
X = data_copy.drop(columns=['HeartDisease'])  # Features
y = data_copy['HeartDisease']  # Target

# Handle categorical variables by setting categories
X['ChestPainType'] = pd.Categorical(X['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
X['RestingECG'] = pd.Categorical(X['RestingECG'], categories=['Normal', 'ST', 'LVH'])
X['ST_Slope'] = pd.Categorical(X['ST_Slope'], categories=['Up', 'Flat', 'Down'])

# Apply one-hot encoding to categorical variables
X = pd.get_dummies(X, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

# Scale numerical features for better model performance
scaler = StandardScaler()  # Initialize StandardScaler
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)  # 80-20 split

# Initialize the Random Forest model with specific parameters
rf_model = RandomForestClassifier(
    n_estimators=250,  # Number of trees
    criterion='entropy',  # Use entropy to measure splits
    max_depth=10,  # Limit the depth of the trees
    min_samples_split=10,  # Minimum samples required to split
    min_samples_leaf=4,  # Minimum samples per leaf node
    max_features='sqrt',  # Number of features to consider per split
    random_state=101,  # Ensure reproducibility
    bootstrap=True  # Use bootstrapping
)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Test if the dataset was loaded correctly
def test_data_loading(data):
    assert data.shape[0] > 0, "Data loading failed, no rows found."
    assert 'HeartDisease' in data.columns, "'HeartDisease' column not found in dataset."
    print("Data loaded successfully!")

# Test if preprocessing steps were applied correctly
def test_data_preprocessing(data_copy):
    assert (data_copy['Cholesterol'] == 0).sum() == 0, "Cholesterol has zero values that weren't dropped."
    print("Data preprocessing completed successfully!")

# Test one-hot encoding of categorical variables
def test_categorical_handling(X):
    assert 'Sex_F' in X.columns and 'Sex_M' in X.columns, "Sex column one-hot encoding failed."
    assert 'ChestPainType_ATA' in X.columns, "ChestPainType one-hot encoding failed."
    print("Categorical handling successful!")

# Test if numerical features were scaled correctly
def test_scaling(X, data):
    original_scaling = data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']].describe()
    assert original_scaling.loc['mean'].isnull().sum() == 0, "Scaling didn't work as expected."
    print("Feature scaling successful!")

# Test if the model was initialized with the correct parameters
def test_model_initialization(rf_model):
    assert rf_model.n_estimators == 250, "Random Forest model initialization failed (n_estimators)."
    assert rf_model.criterion == 'entropy', "Random Forest model initialization failed (criterion)."
    print("Model initialization successful!")

# Test if the model can be trained on the training data
def test_model_training(rf_model, X_train, y_train):
    assert rf_model.fit(X_train, y_train) is not None, "Model fitting failed."
    print("Model trained successfully!")

# Test if predictions are made correctly
def test_model_prediction(rf_model, X_test, y_test):
    y_pred = rf_model.predict(X_test)
    assert len(y_pred) == len(y_test), "Prediction length mismatch."
    print("Model prediction successful!")

# Test evaluation metrics such as accuracy and classification report
def test_evaluation(rf_model, X_test, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    assert 0 <= accuracy <= 1, f"Accuracy score is out of bounds: {accuracy}"
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    report = classification_report(y_test, y_pred)
    assert 'precision' in report, "Classification report does not contain precision."
    print("Evaluation metrics printed successfully!")

# Test exporting model and related objects to files
def test_export_model(rf_model, scaler, X):
    # Export model, scaler, and feature names
    with open('rf_model.pkl', 'wb') as model_file:
        pickle.dump(rf_model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open('feature_names.pkl', 'wb') as feature_file:
        pickle.dump(X.columns.tolist(), feature_file)
    
    # Verify that the files were created successfully
    assert os.path.exists('rf_model.pkl'), "Model file not saved."
    assert os.path.exists('scaler.pkl'), "Scaler file not saved."
    assert os.path.exists('feature_names.pkl'), "Feature names file not saved."
    print("Model and related files exported successfully!")

# Run all integration tests to ensure the entire pipeline works
def run_integration_tests():
    test_data_loading(data)
    test_data_preprocessing(data_copy)
    test_categorical_handling(X)
    test_scaling(X, data)
    test_model_initialization(rf_model)
    test_model_training(rf_model, X_train, y_train)
    test_model_prediction(rf_model, X_test, y_test)
    y_pred = rf_model.predict(X_test)  # Generate predictions
    test_evaluation(rf_model, X_test, y_test, y_pred)
    test_export_model(rf_model, scaler, X)

# Run all defined tests
run_integration_tests()
