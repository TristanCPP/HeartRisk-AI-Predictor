import unittest
import joblib
import pandas as pd
from main_alt import preprocess_user_data, risk_category 

class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load model, scaler, and feature names
        cls.rf_model = joblib.load('rf_model.pkl')
        cls.scaler = joblib.load('scaler.pkl')
        cls.feature_names = joblib.load('feature_names.pkl')

    def test_preprocess_user_data(self):
        # Sample user input
        user_data = {
            'Age': 55,
            'Sex': 'M',
            'ChestPainType': 'ATA',
            'RestingBP': 140,
            'Cholesterol': 220,
            'FastingBS': 1,
            'RestingECG': 'Normal',
            'MaxHR': 150,
            'ExerciseAngina': 'Y',
            'Oldpeak': 1.5,
            'ST_Slope': 'Up'
        }

        # Process the data
        user_df = preprocess_user_data(user_data, self.feature_names, self.scaler)

        # Check if columns match feature_names
        self.assertTrue(all(col in user_df.columns for col in self.feature_names))
        self.assertEqual(user_df.shape[1], len(self.feature_names))

    def test_risk_category(self):
        # Test risk categories
        self.assertEqual(risk_category(0.1), "Low Risk")
        self.assertEqual(risk_category(0.3), "Slight Risk")
        self.assertEqual(risk_category(0.5), "Moderate Risk")
        self.assertEqual(risk_category(0.7), "High Risk")
        self.assertEqual(risk_category(0.9), "Extreme Risk")

    def test_model_prediction(self):
        # Sample user input
        user_data = {
            'Age': 55,
            'Sex': 'M',
            'ChestPainType': 'ATA',
            'RestingBP': 140,
            'Cholesterol': 220,
            'FastingBS': 1,
            'RestingECG': 'Normal',
            'MaxHR': 150,
            'ExerciseAngina': 'Y',
            'Oldpeak': 1.5,
            'ST_Slope': 'Up'
        }

        # Process user data
        user_df = preprocess_user_data(user_data, self.feature_names, self.scaler)

        # Predict risk probability
        risk_probability = self.rf_model.predict_proba(user_df)[0][1]
        self.assertTrue(0 <= risk_probability <= 1)

if __name__ == '__main__':
    unittest.main()
