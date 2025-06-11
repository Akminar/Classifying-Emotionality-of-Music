import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Load test data
x_test = pd.read_csv("../data/processed_data/test_features.csv")
y_test = pd.read_csv("../data/processed_data/test_labels.csv").squeeze()

# Load the trained model
clf = joblib.load("../models/decision_tree_model.pkl")

# Predict and evaluate
y_pred = clf.predict(x_test)
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred))