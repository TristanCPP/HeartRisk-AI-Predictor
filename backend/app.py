import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response
import xml.etree.ElementTree as ET

app = Flask(__name__)

# Load trained Random Forest model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler used during training
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the feature names (to ensure correct input order)
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Function to preprocess user data (matches local test logic)
def preprocess_user_data(user_data):
    """
    Convert user input into a format matching the training data.
    """
    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])

    # Apply categorical encoding
    user_df['ChestPainType'] = pd.Categorical(user_df['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
    user_df['ST_Slope'] = pd.Categorical(user_df['ST_Slope'], categories=['Up', 'Flat', 'Down'])

    # One-hot encode categorical variables
    user_df = pd.get_dummies(user_df, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'ST_Slope'], dtype=int)

    # Add missing columns
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0

    # Reorder columns
    user_df = user_df[feature_names]

    # Scale numerical features
    user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(
        user_df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
    )

    return user_df


@app.route("/predict", methods=["POST"])
def predict():
    try:
        xml_data = request.data.decode("utf-8")
        root = ET.fromstring(xml_data)

        user_data = {
            'Age': int(root.find("Age").text),
            'RestingBP': int(root.find("RestingBP").text),
            'Cholesterol': int(root.find("Cholesterol").text),
            'MaxHR': int(root.find("MaxHR").text),
            'Oldpeak': float(root.find("Oldpeak").text),
            'Sex': root.find("Sex").text,
            'ExerciseAngina': root.find("ExerciseAngina").text,
            'ChestPainType': root.find("ChestPainType").text,
            'ST_Slope': root.find("ST_Slope").text
        }

        input_df = preprocess_user_data(user_data)
        #print("Final Processed Data for Prediction (Flask):\n", input_df)

        risk_score = model.predict_proba(input_df)[0, 1] * 100  # Convert to percentage

        print(f"📊 Risk Score from Model: {risk_score:.2f}%")  # Debugging Line

        response_xml = f"""<HeartRiskResponse><RiskScore>{risk_score:.2f}%</RiskScore></HeartRiskResponse>"""
        return Response(response_xml, mimetype="application/xml")

    except Exception as e:
        print(f"⚠️ Error processing request: {e}")
        return f"<Error>{e}</Error>", 400



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

