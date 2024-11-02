import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the new dataset
data = pd.read_csv('heart_disease_data.csv')
data.info()

# Map categorical columns to numerical values
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
data['ChestPainType'] = data['ChestPainType'].map({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3})
data['RestingECG'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
data['ExerciseAngina'] = data['ExerciseAngina'].map({'N': 0, 'Y': 1})
data['ST_Slope'] = data['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})


# Rename the target column for consistency
data.rename(columns={'HeartDisease': 'target'}, inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Split features (X) and target (y)
X = data.drop(columns='target')
y = data['target']

# Standardize features (scale numerical data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier with optimized parameters for better accuracy
rf_model = RandomForestClassifier(
    n_estimators=250,          
    criterion='entropy', 
    max_depth=10,              # Limit the depth of each tree to prevent overfitting
    min_samples_split=10,      # Require a minimum of 10 samples to split a node
    min_samples_leaf=4,        # Require at least 4 samples at each leaf node
    max_features='sqrt',       # Consider a subset of features (log2) for each split
    random_state=42,
    bootstrap=True,            # to reduce variance
)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%')

# Train the model on the entire training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print a detailed classification report
print(classification_report(y_test, y_pred))

# Get the predicted probabilities for each sample in the test set
y_probs = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (heart disease)

# Define risk tiers based on probability thresholds
def categorize_risk(prob):
    if prob < 0.2:
        return 'Low Risk (Green)'
    elif prob < 0.4:
        return 'Slight Risk (Yellow)'
    elif prob < 0.6:
        return 'Moderate Risk (Orange)'
    elif prob < 0.8:
        return 'High Risk (Dark Orange)'
    else:
        return 'Extreme Risk (Bright Red)'

# Apply the categorization to the predicted probabilities
risk_categories = [categorize_risk(prob) for prob in y_probs]

# Print the first few results
print(risk_categories[:10])

# Print the probabilities for the test data
print(y_probs)

# (Optional) Plot the feature correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

