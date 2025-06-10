import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt

# Train the decision tree
x_train = pd.read_csv("../data/processed_data/train_features.csv")
x_test = pd.read_csv("../data/processed_data/test_features.csv")
y_train = pd.read_csv("../data/processed_data/train_labels.csv").squeeze()
y_test = pd.read_csv("../data/processed_data/test_labels.csv").squeeze()

clf = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

# Find the best parameters w/ grid search
param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5],
}

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(x_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Evaluate with best model
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(x_test)

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Distracting", "Focus-Friendly"]))

# Print the feature importances
importances = best_clf.feature_importances_
feature_names = x_train.columns

sorted_indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[sorted_indices])
plt.xticks(range(len(importances)), feature_names[sorted_indices])
plt.tight_layout()
plt.show()

# Print the decision tree
plt.figure(figsize=(20, 10))
plot_tree(best_clf, 
          feature_names=x_train.columns, 
          class_names=["Distracting", "Focus-Friendly"],
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.show()

# Save the model
joblib.dump(clf, "../data/models/decision_tree_model.pkl")