import unittest
from unittest.mock import patch, MagicMock
from main import feature_names, preprocess_and_predict, generate_health_suggestions, validate_input
from tkinter import StringVar, Entry

class TestPredictionFunctions(unittest.TestCase):

    @patch('joblib.load')
    def test_preprocess_and_predict(self, mock_load):
        # Mock the model and scaler loading
        mock_rf_model = MagicMock()
        mock_scaler = MagicMock()
        mock_load.side_effect = [mock_rf_model, mock_scaler, feature_names]

        # Mock model prediction to simulate high risk
        mock_rf_model.predict_proba.return_value = [[0.1, 0.9]]  # Simulate high-risk probability

        # Mock scaler transformation
        mock_scaler.transform.return_value = [[0.5, 0.5, 0.5, 0.5, 0.5]]

        data = {
            'Age': 67,
            'Sex': 'M',
            'ChestPainType': 'ASY',
            'RestingBP': 140,
            'Cholesterol': 200,
            'FastingBS': 1,
            'RestingECG': 'Normal',
            'MaxHR': 150,
            'ExerciseAngina': 'Y',
            'Oldpeak': 2.0,
            'ST_Slope': 'Up'
        }

        probability, category, suggestions = preprocess_and_predict(data)

        self.assertEqual(category, "High Risk")  # Expect high risk
        self.assertGreater(probability, 0.61)  # Validate probability
        self.assertTrue(any("cholesterol" in s for s in suggestions))  # Partial match


    def test_generate_health_suggestions(self):
        data = {
            'Cholesterol': 250,
            'RestingBP': 150,
            'FastingBS': 1,
            'MaxHR': 100,
            'ExerciseAngina': 'N',
            'Oldpeak': 2.5
        }
        suggestions = generate_health_suggestions(data)
        self.assertTrue(any("critically high" in s for s in suggestions))
        self.assertTrue(any("hypertensive range" in s for s in suggestions))

@patch('tkinter.Entry.get')
@patch('tkinter.StringVar.get')
def test_validate_input(self, mock_get_string, mock_get_entry):
    # Valid input simulation
    mock_get_entry.side_effect = ['50', '120', '200', '150', '2', '1']  # Add values for all numeric fields
    mock_get_string.side_effect = ['M', 'ATA', 'Normal', 'N', 'Up']  # Add values for all dropdowns
    result = validate_input()
    self.assertIsNotNone(result)
    self.assertEqual(result['Age'], 50)

    # Invalid input simulation (Empty fields)
    mock_get_entry.side_effect = [''] * 5  # Empty numeric fields
    mock_get_string.side_effect = [''] * 5  # Empty dropdowns
    result = validate_input()
    self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
