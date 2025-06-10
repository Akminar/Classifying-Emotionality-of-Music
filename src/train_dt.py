import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, learning_curve
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Train the decision tree
x = pd.read_csv("../data/processed_data/train_features.csv")
y = pd.read_csv("../data/processed_data/train_labels.csv").squeeze()

clf = DecisionTreeClassifier(
    class_weight = "balanced",
    random_state = 42,
    max_depth = 5,
    min_samples_split = 5,
    min_samples_leaf = 2
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get the learning curves
train_sizes, train_scores, val_scores = learning_curve(
    clf, x, y, cv=cv, scoring="f1_weighted", train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot the learning curves
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

# Cross validation
f1_scores = cross_val_score(clf, x, y, cv=cv, scoring='f1_weighted')
print("F1 Weighted Scores Per Fold:", f1_scores)
print("Mean F1 Weighted Score:", np.mean(f1_scores))

y_pred_cv = cross_val_predict(clf, x, y, cv=cv)

clf.fit(x, y)

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred_cv))

# Print the classification report
print("\nClassification Report:")
print(classification_report(y, y_pred_cv, target_names=["Distracting", "Focus-Friendly"]))

# Print the feature importances
importances = clf.feature_importances_
feature_names = x.columns

sorted_indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[sorted_indices])
plt.xticks(range(len(importances)), feature_names[sorted_indices])
plt.tight_layout()
plt.show()

# Print the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=x.columns, 
          class_names=["Distracting", "Focus-Friendly"],
          filled=True, 
          rounded=True, 
          fontsize=8)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.show()

# Save the model
joblib.dump(clf, "../data/models/decision_tree_model.pkl")