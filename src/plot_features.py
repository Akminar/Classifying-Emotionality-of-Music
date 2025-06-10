import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import learning_curve
import joblib
import numpy as np
import os

# Load trained model
clf = joblib.load("../models/decision_tree_model.pkl")

# Load datasets
x = pd.read_csv("../data/processed_data/train_features.csv")
y = pd.read_csv("../data/processed_data/train_labels.csv").squeeze()

# Load merged dataset to get feature names
merged_df = pd.read_csv("../data/merged_features.csv")

# Isolate feature names
non_feature_cols = ["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std", "label"]
feature_names = [col for col in merged_df.columns if col not in non_feature_cols]

"""
# Plot decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feature_names, class_names=["Distracting", "Focus-Friendly"], 
          filled=True, fontsize=10)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()

# Plot top 10 feature importances
importances = clf.feature_importances_
indices = importances.argsort()[::-1]  # Sort descending

top_n = 10
top_features = [feature_names[i] for i in indices[:top_n]]

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[indices[:top_n]][::-1])
plt.yticks(range(top_n), top_features[::-1])
plt.xlabel("Feature Importance")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

# Confusion Matrix
y_pred = clf.predict(x)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Distracting", "Focus-Friendly"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Classification Report
print(classification_report(y, y_pred, target_names=["Distracting", "Focus-Friendly"]))
"""
# Learning Curves
train_sizes, train_scores, val_scores = learning_curve(
    clf, x, y, cv=5, scoring="f1_weighted", train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training score", marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)

plt.plot(train_sizes, val_mean, label="Validation score", marker='s')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

plt.title("Learning Curve: Decision Tree")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score (weighted)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
