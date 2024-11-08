# NOT WORKING / ISSUES WITH VS CODE COMPATIBILITY

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load and preprocess the dataset
data = pd.read_csv('data/heart_disease_data.csv')
processed_data = data[data['Cholesterol'] != 0]
X = processed_data.drop(columns=['HeartDisease'])
y = processed_data['HeartDisease']

# One-hot encode categorical features and scale numerical features
X['ChestPainType'] = pd.Categorical(X['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
X['RestingECG'] = pd.Categorical(X['RestingECG'], categories=['Normal', 'ST', 'LVH'])
X['ST_Slope'] = pd.Categorical(X['ST_Slope'], categories=['Up', 'Flat', 'Down'])
X = pd.get_dummies(X, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

scaler = StandardScaler()
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Define the function to create the neural network model
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train multiple neural networks (ensemble)
n_ensembles = 5
ensemble_predictions = []

for i in range(n_ensembles):
    print(f"Training model {i + 1}/{n_ensembles}...")
    model = create_model()
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.1)
    predictions = model.predict(X_test).flatten()
    ensemble_predictions.append(predictions)

# Average the predictions across the ensemble
ensemble_predictions = np.array(ensemble_predictions)
average_predictions = np.mean(ensemble_predictions, axis=0)
final_predictions = (average_predictions > 0.5).astype(int)  # Convert probabilities to binary labels

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, final_predictions)
print("Ensemble Neural Network Accuracy:", accuracy)
print("Ensemble Neural Network Classification Report:\n", classification_report(y_test, final_predictions))
