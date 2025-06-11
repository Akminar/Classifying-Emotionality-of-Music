import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

torch.manual_seed(42)

# Load data
x = pd.read_csv("../data/processed_data/train_features.csv").values
y = pd.read_csv("../data/processed_data/train_labels.csv").squeeze().values

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

epochs = 30
lr = 0.001
k_folds = 5

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []
best_model = None
best_f1 = 0.0
train_f1_per_epoch = []
val_f1_per_epoch = []

for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    print(f"\\nFold {fold + 1}/{k_folds}")
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = MLP(input_dim=x.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_f1_history = []
    val_f1_history = []

    for epoch in tqdm(range(epochs), desc=f"Training Fold {fold+1}"):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_preds = model(x_train)
            val_preds = model(x_val)
            train_f1 = f1_score(y_train, torch.argmax(train_preds, dim=1), average="weighted")
            val_f1 = f1_score(y_val, torch.argmax(val_preds, dim=1), average="weighted")

        train_f1_history.append(train_f1)
        val_f1_history.append(val_f1)

    print(f"Final Fold {fold + 1} Validation F1 score: {val_f1:.4f}")
    fold_results.append(val_f1)

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model = model

    train_f1_per_epoch.append(train_f1_history)
    val_f1_per_epoch.append(val_f1_history)

os.makedirs("../models", exist_ok=True)
torch.save(best_model.state_dict(), "../models/mlp_model.pt")

avg_train_f1 = np.mean(train_f1_per_epoch, axis=0)
avg_val_f1 = np.mean(val_f1_per_epoch, axis=0)

os.makedirs("../figures", exist_ok=True)
plt.plot(avg_train_f1, label="Train F1")
plt.plot(avg_val_f1, label="Validation F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("MLP Cross-Validation Learning Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/mlp_cv_learning_curve.png")
plt.close()

print(f"\\nAverage Validation F1 across folds: {np.mean(fold_results):.4f}")