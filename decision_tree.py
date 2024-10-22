import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load the dataset
data = pd.read_csv('Heart_disease_cleveland_new.csv')

# Define features (independent variables) and the target (dependent variable)
features = data.drop('target', axis=1)
target = data['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Test the classifier
y_pred = clf.predict(X_test)

# Print accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# # Function to take user input and predict heart disease
# def predict_heart_disease():
#     print("Please enter the following health information:")
#     age = int(input("Age: "))
#     sex = int(input("Sex (1=Male, 0=Female): "))
#     cp = int(input("Chest pain type (0-3): "))
#     trestbps = int(input("Resting blood pressure: "))
#     chol = int(input("Cholesterol level: "))
#     fbs = int(input("Fasting blood sugar (1=True, 0=False): "))
#     restecg = int(input("Resting ECG results (0-2): "))
#     thalach = int(input("Maximum heart rate achieved: "))
#     exang = int(input("Exercise induced angina (1=Yes, 0=No): "))
#     oldpeak = float(input("ST depression induced by exercise: "))
#     slope = int(input("Slope of the peak exercise ST segment (0-2): "))
#     ca = int(input("Number of major vessels colored by fluoroscopy (0-3): "))
#     thal = int(input("Thalassemia (1=Normal, 2=Fixed defect, 3=Reversible defect): "))
    
#     user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    
#     # Predict based on user input
#     prediction = clf.predict(user_data)
    
#     if prediction == 1:
#         print("Prediction: You are at risk of heart disease.")
#     else:
#         print("Prediction: You are not at risk of heart disease.")

# # Call the prediction function
# predict_heart_disease()

