import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# Load the new dataset
data = pd.read_csv('data/heart_disease_data.csv')

# Make a copy of the original dataset
data_copy = data.copy(deep=True)

# Dropping rows where Cholesterol is 0
data_copy = data_copy[(data_copy['Cholesterol'] != 0)]

# Define features (X) and target (y)
X = data_copy.drop(columns=['HeartDisease'])
y = data_copy['HeartDisease']

# Define categorical data with all possible categories
X['ChestPainType'] = pd.Categorical(X['ChestPainType'], categories=['ATA', 'NAP', 'ASY', 'TA'])
X['RestingECG'] = pd.Categorical(X['RestingECG'], categories=['Normal', 'ST', 'LVH'])
X['ST_Slope'] = pd.Categorical(X['ST_Slope'], categories=['Up', 'Flat', 'Down'])

# One-Hot Encode non-binary categorical variables without dropping any category
X = pd.get_dummies(X, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG', 'ST_Slope'], dtype=int)

# Scale numerical features
scaler = StandardScaler()
X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
    X[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Initialize the Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train the model on the training data
nb_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Naive Bayes Accuracy: {accuracy * 100:.2f}%')

# Print a detailed classification report
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred))





# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", conf_matrix)


# # Perform cross-validation
# cv_scores = cross_val_score(nb_model, X, y, cv=5, scoring='accuracy')
# print(f'Cross-Validation Accuracy Scores: {cv_scores}')
# print(f'Mean Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%')
