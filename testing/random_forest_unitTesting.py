import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Unit test class for testing the heart disease prediction model
class TestHeartDiseaseModel(unittest.TestCase):

    @patch('pandas.read_csv')  # Mock the `read_csv` function to simulate data loading
    def test_data_loading(self, mock_read_csv):
        # Mocked dataset returned by `read_csv`
        mock_read_csv.return_value = pd.DataFrame({
            'Age': [63, 37],
            'Sex': ['M', 'M'],
            'ChestPainType': ['ATA', 'NAP'],
            'RestingBP': [145, 130],
            'Cholesterol': [233, 250],
            'MaxHR': [150, 180],
            'Oldpeak': [2.3, 1.5],
            'HeartDisease': [1, 0]
        })
        # Simulate loading the dataset
        data = pd.read_csv('datasets/heart_disease_data.csv')
        # Assert the dataset has the expected number of rows
        self.assertEqual(data.shape[0], 2)
        # Assert important columns exist
        self.assertTrue('Age' in data.columns)
        self.assertTrue('HeartDisease' in data.columns)

    # Test for preprocessing the data
    def test_data_preprocessing(self):
        # Create a small sample DataFrame for testing preprocessing
        data_copy = pd.DataFrame({
            'Age': [63, 37],
            'Cholesterol': [233, 0],  # One row has invalid cholesterol
            'Sex': ['M', 'F']
        })
        # Remove rows where Cholesterol = 0
        data_copy = data_copy[data_copy['Cholesterol'] != 0]
        # Assert only valid rows remain
        self.assertEqual(data_copy.shape[0], 1)
        
        # Convert 'Sex' column to categorical type
        data_copy['Sex'] = pd.Categorical(data_copy['Sex'], categories=['M', 'F'])
        # Perform one-hot encoding on 'Sex' column
        data_copy = pd.get_dummies(data_copy, columns=['Sex'], dtype=int)
        # Assert one-hot encoding worked correctly
        self.assertIn('Sex_M', data_copy.columns)

    @patch('sklearn.model_selection.train_test_split')  # Mock train_test_split function
    def test_model_training(self, mock_train_test_split):
        # Mocked train-test split data
        mock_train_test_split.return_value = (
            pd.DataFrame({'Age': [63, 37]}),  # Training features
            pd.DataFrame({'Age': [45]}),  # Test features
            pd.Series([1, 0]),  # Training labels
            pd.Series([1])  # Test labels
        )
        # Initialize Random Forest model
        rf_model = RandomForestClassifier(n_estimators=250, random_state=101)
        # Get mock train-test split
        X_train, X_test, y_train, y_test = mock_train_test_split.return_value
        # Train the model
        rf_model.fit(X_train, y_train)
        # Assert the model has been trained (check for attribute)
        self.assertTrue(hasattr(rf_model, 'feature_importances_'))

    @patch('sklearn.ensemble.RandomForestClassifier.predict')  # Mock the `predict` method of RandomForestClassifier
    def test_model_predictions(self, mock_predict):
        # Mocked predictions returned by `predict`
        mock_predict.return_value = np.array([1, 0])  # Predicted labels
        # Initialize Random Forest model
        rf_model = RandomForestClassifier()
        # Create test feature data
        X_test = pd.DataFrame({'Age': [63, 37]})
        # Get predictions
        y_pred = rf_model.predict(X_test)
        # Assert predictions match the mocked values
        self.assertEqual(y_pred.tolist(), [1, 0])

    # Test evaluation metrics (accuracy and classification report)
    def test_model_evaluation(self):
        # Sample ground truth and predicted labels
        y_test = np.array([1, 0])
        y_pred = np.array([1, 0])
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Assert the accuracy is as expected
        self.assertEqual(accuracy, 1.0)  # 100% accuracy for matching labels
        # Generate a classification report
        report = classification_report(y_test, y_pred)
        # Assert the report contains the accuracy metric
        self.assertIn('accuracy', report)

# Run all tests in the class
if __name__ == '__main__':
    unittest.main()
