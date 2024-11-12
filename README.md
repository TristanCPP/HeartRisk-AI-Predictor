# Machine Learning Powered Heart Disease Risk Prediction

![Project Banner](https://img.shields.io/badge/HeartRiskAI-Predictor-blue)

## Overview
The **HeartRisk AI Predictor** is a machine learning-based application that predicts the likelihood of coronary heart disease (CHD) based on user-provided health data. The system is built using a Random Forest model and provides dynamic risk categorization (e.g., Low, Moderate, High Risk) and actionable feedback for users. It is designed as a backend system with plans to integrate a mobile application frontend.

## Features
- **Machine Learning Model**: A Random Forest classifier optimized for accuracy (~92.67%) on a curated heart disease dataset.
- **Risk Prediction**: Provides a CHD risk probability based on user health metrics.
- **Risk Categorization**: Classifies users into five risk tiers: Low Risk, Slight Risk, Moderate Risk, High Risk, Extreme Risk.
- **Customizable Inputs**: Supports real-time user inputs with preprocessing and scaling for prediction.
- **Unit Testing**: Includes test coverage for key functionalities like preprocessing, prediction, and accuracy validation.
- **Data Visualizations**: Demonstrates dataset and model performance using heatmaps, distribution plots, feature importance, and confusion matrices.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

---

## Technologies Used
- **Python 3.12**: Core language for development.
- **Libraries**:
  - Machine Learning: `scikit-learn`
  - Data Manipulation: `pandas`, `numpy`
  - Visualizations: `matplotlib`, `seaborn`, `shap`
  - Backend Utilities: `joblib`, `pickle`
- **Random Forest Classifier**: Primary model for CHD prediction.
- **Unit Testing**: `unittest` framework.

---

## Project Structure

Heart Disease Risk Prediction/
  - data/
      - heart_disease_data.csv        # Main dataset
  - models/
      - (Main) random_forest_updated.py      # Random Forest training script
  - visualizations/
      - visualize_data_model.py       # Visualization scripts for better understanding datasets and machine learning models
  - tests/
      - test_main.py                  # Unit tests for main functionality
      - test_updated_random_forest.py # Unit tests for Random Forest model
  - main.py                           # Hardcoded Inputs Application Script that outputs risk score and risk category based on given inputs
  - main_inputs.py                    # Main Application Script that simulates user inputs and input error handling and outputs the risk score and risk category
  - rf_model.pkl                      # Saved Random Forest model
  - README.md                         # Project documentation



---

## Future Enhancements:
- Frontend Development: Integrate a mobile application using React Native for user interaction.
- Real-Time Feedback: Provide lifestyle recommendations to reduce CHD risk.
- Model Optimization: Further iterate and improve the current machine learning model.
- Deployment: Host the application backend and frontend on a cloud platform for live usage.

---

## Contributors:
Team Members: Tristan Garner, Muhsen AbuMuhsen, Chase Lillard
