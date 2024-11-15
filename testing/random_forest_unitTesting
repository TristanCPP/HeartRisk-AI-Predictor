import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class TestHeartDiseaseModel(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_data_loading(self, mock_read_csv):
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
        data = pd.read_csv('datasets/heart_disease_data.csv')
        self.assertEqual(data.shape[0], 2)
        self.assertTrue('Age' in data.columns)
        self.assertTrue('HeartDisease' in data.columns)
    
    def test_data_preprocessing(self):
        data_copy = pd.DataFrame({
            'Age': [63, 37],
            'Cholesterol': [233, 0],
            'Sex': ['M', 'F']
        })
        # Remove rows with Cholesterol = 0
        data_copy = data_copy[data_copy['Cholesterol'] != 0]
        self.assertEqual(data_copy.shape[0], 1)
        
        # Convert 'Sex' column to categorical
        data_copy['Sex'] = pd.Categorical(data_copy['Sex'], categories=['M', 'F'])
        # Perform one-hot encoding on 'Sex'
        data_copy = pd.get_dummies(data_copy, columns=['Sex'], dtype=int)
        self.assertIn('Sex_M', data_copy.columns)

    @patch('sklearn.model_selection.train_test_split')
    def test_model_training(self, mock_train_test_split):
        mock_train_test_split.return_value = (
            pd.DataFrame({'Age': [63, 37]}), 
            pd.DataFrame({'Age': [45]}),
            pd.Series([1, 0]),
            pd.Series([1])
        )
        rf_model = RandomForestClassifier(n_estimators=250, random_state=101)
        X_train, X_test, y_train, y_test = mock_train_test_split.return_value
        rf_model.fit(X_train, y_train)
        self.assertTrue(hasattr(rf_model, 'feature_importances_'))

    @patch('sklearn.ensemble.RandomForestClassifier.predict')
    def test_model_predictions(self, mock_predict):
        mock_predict.return_value = np.array([1, 0])
        rf_model = RandomForestClassifier()
        X_test = pd.DataFrame({'Age': [63, 37]})
        y_pred = rf_model.predict(X_test)
        self.assertEqual(y_pred.tolist(), [1, 0])

    def test_model_evaluation(self):
        y_test = np.array([1, 0])
        y_pred = np.array([1, 0])
        accuracy = accuracy_score(y_test, y_pred)
        self.assertEqual(accuracy, 1.0)
        report = classification_report(y_test, y_pred)
        self.assertIn('accuracy', report)

if __name__ == '__main__':
    unittest.main()
