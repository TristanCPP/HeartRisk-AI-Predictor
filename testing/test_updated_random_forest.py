import unittest
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class TestRandomForest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load pre-trained model and feature names
        with open('rf_model.pkl', 'rb') as model_file:
            cls.rf_model = pickle.load(model_file)
        with open('feature_names.pkl', 'rb') as feature_file:
            cls.feature_names = pickle.load(feature_file)

    def test_model_training(self):
        # Ensure the model is a RandomForestClassifier
        self.assertIsInstance(self.rf_model, RandomForestClassifier)

    def test_model_accuracy(self):
        # Load a small subset of data for testing
        test_data = {
            'Age': [40, 50],
            'RestingBP': [130, 150],
            'Cholesterol': [200, 300],
            'MaxHR': [170, 140],
            'Oldpeak': [1.2, 0.8],
            'FastingBS': [0, 1],
            'Sex_M': [1, 0],
            'Sex_F': [0, 1],
            'ChestPainType_ATA': [1, 0],
            'ChestPainType_NAP': [0, 1],
            'ChestPainType_ASY': [0, 0],
            'ChestPainType_TA': [0, 0],
            'RestingECG_Normal': [1, 0],
            'RestingECG_ST': [0, 1],
            'RestingECG_LVH': [0, 0],
            'ST_Slope_Up': [1, 0],
            'ST_Slope_Flat': [0, 1],
            'ST_Slope_Down': [0, 0]
        }
        
        X_test = pd.DataFrame(test_data)

        # Ensure the test data columns match the model's feature names
        for col in self.feature_names:
            if col not in X_test.columns:
                X_test[col] = 0  # Add missing columns with a default value of 0

        # Reorder columns to match training data order
        X_test = X_test[self.feature_names]

        # Define true labels for test cases
        y_test = [0, 1]

        # Predict and evaluate
        y_pred = self.rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Ensure predictions have at least 50% accuracy
        self.assertGreater(accuracy, 0.7)

if __name__ == '__main__':
    unittest.main()
