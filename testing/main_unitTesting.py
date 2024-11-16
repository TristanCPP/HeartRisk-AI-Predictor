import unittest
from unittest.mock import patch, MagicMock
from main import feature_names, preprocess_and_predict, generate_health_suggestions, validate_input
from tkinter import StringVar, Entry

# Unit test class for prediction functions
class TestPredictionFunctions(unittest.TestCase):

    @patch('joblib.load')  # Mock the `joblib.load` function to prevent loading actual model files
    def test_preprocess_and_predict(self, mock_load):
        # Mock the model and scaler objects
        mock_rf_model = MagicMock()  # Mocked Random Forest model
        mock_scaler = MagicMock()  # Mocked scaler for feature scaling
        mock_load.side_effect = [mock_rf_model, mock_scaler, feature_names]  # Simulate model, scaler, and feature_names loading

        # Mock the model's prediction output to simulate a high-risk scenario
        mock_rf_model.predict_proba.return_value = [[0.1, 0.9]]  # Second value represents high-risk probability

        # Mock the scaler's transformation output to simulate scaled feature values
        mock_scaler.transform.return_value = [[0.5, 0.5, 0.5, 0.5, 0.5]]  # Simulate normalized data

        # Sample input data for prediction
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

        # Call the function to preprocess data and predict risk
        probability, category, suggestions = preprocess_and_predict(data)

        # Assert that the risk category is correctly identified as high risk
        self.assertEqual(category, "High Risk")  # Expect "High Risk" category
        self.assertGreater(probability, 0.61)  # Validate that probability is above the high-risk threshold
        self.assertTrue(any("cholesterol" in s for s in suggestions))  # Check if cholesterol-related suggestion exists

    # Test for generating health suggestions based on input data
    def test_generate_health_suggestions(self):
        # Sample input data for health suggestions
        data = {
            'Cholesterol': 250,  # High cholesterol
            'RestingBP': 150,  # High resting blood pressure
            'FastingBS': 1,  # Elevated fasting blood sugar
            'MaxHR': 100,  # Low maximum heart rate
            'ExerciseAngina': 'N',  # No exercise-induced angina
            'Oldpeak': 2.5  # ST depression indicating stress
        }

        # Generate health suggestions based on the data
        suggestions = generate_health_suggestions(data)

        # Validate the suggestions contain expected warnings or advice
        self.assertTrue(any("critically high" in s for s in suggestions))  # Check for critical warnings
        self.assertTrue(any("hypertensive range" in s for s in suggestions))  # Check for hypertensive warning

# Unit test for validating user input in the GUI
@patch('tkinter.Entry.get')  # Mock the `Entry.get` method for numeric input fields
@patch('tkinter.StringVar.get')  # Mock the `StringVar.get` method for dropdown input fields
def test_validate_input(self, mock_get_string, mock_get_entry):
    # Simulate valid user input
    mock_get_entry.side_effect = ['50', '120', '200', '150', '2', '1']  # Provide valid numeric inputs
    mock_get_string.side_effect = ['M', 'ATA', 'Normal', 'N', 'Up']  # Provide valid dropdown selections
    result = validate_input()  # Call the input validation function
    self.assertIsNotNone(result)  # Assert that validation returns a non-None result
    self.assertEqual(result['Age'], 50)  # Validate that the age field is correctly parsed

    # Simulate invalid input with empty fields
    mock_get_entry.side_effect = [''] * 5  # Simulate empty numeric fields
    mock_get_string.side_effect = [''] * 5  # Simulate empty dropdown selections
    result = validate_input()  # Call the input validation function
    self.assertIsNone(result)  # Assert that validation fails and returns None

# Main entry point for running the tests
if __name__ == '__main__':
    unittest.main()  # Run all defined unit tests
