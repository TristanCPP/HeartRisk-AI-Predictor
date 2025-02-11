import joblib
import pandas as pd
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Load the trained RF model
rf_model = joblib.load("rf_model.pkl")

# Load dataset to get feature names
df = pd.read_csv("data/heart_disease_data_test.csv")

# Remove rows where Cholesterol is 0 (matching preprocessing)
df = df[df["Cholesterol"] != 0]

# Separate features and target variable
X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

# Apply categorical encoding similar to the training script
X["ChestPainType"] = pd.Categorical(X["ChestPainType"], categories=["ATA", "NAP", "ASY", "TA"])
X["ST_Slope"] = pd.Categorical(X["ST_Slope"], categories=["Up", "Flat", "Down"])

# Apply one-hot encoding
X = pd.get_dummies(X, columns=["Sex", "ExerciseAngina", "ChestPainType", "ST_Slope"], dtype=int)

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")
X[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]] = scaler.transform(
    X[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]]
)

# Define the PMML pipeline with the Random Forest model
pipeline = PMMLPipeline([
    ("classifier", rf_model)
])

# Fit the pipeline (needed for PMML conversion)
pipeline.fit(X, y)

# Save the model as a PMML file
sklearn2pmml(pipeline, "heart_disease_model.pmml", with_repr=True)

print("Model successfully exported to PMML!")
