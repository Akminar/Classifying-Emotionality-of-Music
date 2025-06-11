import torch
import pandas as pd
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix

# Load test data
x_test = pd.read_csv("../data/processed_data/test_features.csv").values
y_test = pd.read_csv("../data/processed_data/test_labels.csv").values.squeeze()

# Define model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

# Load model
input_size = x_test.shape[1]
model = MLP(input_size)
model.load_state_dict(torch.load("../models/mlp_model.pt"))
model.eval()

# Convert test data to tensor
x_tensor = torch.tensor(x_test, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    outputs = model(x_tensor)
    predictions = torch.argmax(outputs, dim=1).numpy()  # Get predicted class indices

# Evaluate
print("MLP Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["Distracting", "Focus-Friendly"]))
