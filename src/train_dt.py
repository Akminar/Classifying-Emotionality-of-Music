import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, learning_curve
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Model save path
out_path = "../models/decision_tree_model.pkl"

# Train the decision tree
x = pd.read_csv("../data/processed_data/train_features.csv")
y = pd.read_csv("../data/processed_data/train_labels.csv").squeeze()

clf = DecisionTreeClassifier(
    class_weight = "balanced",
    random_state = 42,
    max_depth = 7,
    min_samples_split = 5,
    min_samples_leaf = 5
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Cross validation
f1_scores = cross_val_score(clf, x, y, cv=cv, scoring='f1_weighted')
print("F1 Weighted Scores Per Fold:", f1_scores)
print("Mean F1 Weighted Score:", np.mean(f1_scores))

y_pred_cv = cross_val_predict(clf, x, y, cv=cv)

clf.fit(x, y)

print("Training complete. Saving to ", out_path)

# Save the model
joblib.dump(clf, out_path)