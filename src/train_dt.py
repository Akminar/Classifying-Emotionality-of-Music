import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Train the decision tree
x_train = pd.read_csv("../data/processed_data/train_features.csv")
x_test = pd.read_csv("../data/processed_data/test_features.csv")
y_train = pd.read_csv("../data/processed_data/train_labels.csv").squeeze()
y_test = pd.read_csv("../data/processed_data/test_labels.csv").squeeze()

clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Distracting", "Focus-Friendly"]))

# Print the feature importances
importances = clf.feature_importances_
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
plot_tree(clf, 
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