# Machine Learning Powered Heart Disease Risk Prediction

![Project Banner](https://img.shields.io/badge/HeartRiskAI-Predictor-blue)

## Overview
This project is a machine learning-based application that predicts the likelihood of coronary heart disease (CHD) based on user-provided health data. The system is built using a Random Forest model and provides a current risk score and risk categorization (e.g., Low, Moderate, High Risk) to help users understand their current risk of developing the disease. It is designed as a backend system with plans to integrate a mobile application frontend.

## Features
- **Machine Learning Model**: A Random Forest classifier optimized for accuracy (~92.67%) on a curated heart disease dataset.
- **Risk Prediction**: Provides a CHD risk probability based on user health metrics.
- **Risk Categorization**: Classifies users into five risk tiers: Low Risk, Slight Risk, Moderate Risk, High Risk, Extreme Risk.
- **Customizable Inputs**: Supports user inputs with preprocessing and scaling for prediction.
- **Testing**: Includes test coverage for key functionalities like preprocessing, prediction, and accuracy validation.
- **Data Visualizations**: Demonstrates dataset and model performance using heatmaps, distribution plots, feature importance, and confusion matrices.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

---

## Technologies Used
- **Python 3.12**: Core language for development.
- **Libraries**:
  - Machine Learning: `scikit-learn`
  - Data Manipulation: `pandas`, `numpy`
  - Visualizations: `matplotlib`, `seaborn`, `tkinter`
  - Backend Utilities: `joblib`, `pickle`
- **Random Forest Classifier**: Primary model for CHD prediction.
- **Testing**: `unittest` framework.

---

## Project Structure
Heart Disease Risk Prediction/
  - backend/
      - app.py                              # Beginning of backend with API for integration with frontend (Work in Progress)
  - data/
      - heart_disease_data.csv              # Main dataset
  - models/
      - (Main) random_forest_updated.py     # Random Forest training script
      - All other models that were tested
  - visualizations/
      - visualize_data_model.py             # Visualization scripts for better understanding datasets and machine learning models
      - visualization images 
  - tests/
      - main_integrationTesting.py          # Integration tests for main functionality
      - main_unitTesting.py                 # Unit tests for main functions and edge cases
      - random_forest_integrationTesting.py # Integration testing for random forest model functionality
      - random_forest_unitTesting.py        # Unit tests for random forest functions

  - main_alt.py                             # Hardcoded Inputs Application Script that outputs risk score and risk category based on given inputs (For Testing)
  - main.py                                 # Main Application Script that simulates user inputs and input error handling and outputs the risk score and risk category in a simulated interface
  - rf_model.pkl                            # Saved Random Forest model to used for importing into necessary files
  - scaler.pkl                              # Saved Scaler used for importing into necessary files
  - feature_names.pkl                       # Saved Feature Names in order to ensure that data frames match
  - README.md                               # Project documentation

---

## Testing
The ability to run unit/integration tests is made simple by Python's unittest framework.
To run any of the tests just type 'python -m testing."NAME OF FILE".

For example:
 - "python -m testing.main_unitTesting" will run the Unit Tests built for the main.py file
and output the results of the tests.
- Software testing has been conducting as a whole on main.py

---

## Future Enhancements:
- Frontend Development: Integrate a mobile application using React Native for user interaction.
- Optimize Lifestyle Feedback: We plan to evolve the user feedback to be more dynamic than the current coded solution.
- Model Optimization: Further iterate and improve the current machine learning model.
- Deployment: Host the application backend and frontend on a cloud platform for live usage.

---

## Contributors:
Team Members: Tristan Garner, Muhsen AbuMuhsen, Chase Lillard
