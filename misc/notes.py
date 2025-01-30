# **Risk Tier & HeatMap Code**

## Risk Tiers
# Get the predicted probabilities for each sample in the test set
#   y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class (heart disease)

# Define risk tiers based on probability thresholds
# def categorize_risk(prob):
#     if prob < 0.2:
#         return 'Low Risk (Green)'
#     elif prob < 0.4:
#         return 'Slight Risk (Yellow)'
#     elif prob < 0.6:
#         return 'Moderate Risk (Orange)'
#     elif prob < 0.8:
#         return 'High Risk (Dark Orange)'
#     else:
#         return 'Extreme Risk (Bright Red)'
    
# Apply the categorization to the predicted probabilities
#   risk_categories = [categorize_risk(prob) for prob in y_probs]

# # Print the first few risk categories
# print(risk_categories[:10])

# # Print the predicted probabilities for the test data
# print(y_probs)

## Heatmap
# plt.figure(figsize=(10,8))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title("Feature Correlation Matrix")
# plt.show()


# # Calculate the confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Alternatively, use a visualization
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No CHD', 'CHD'])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.show()
