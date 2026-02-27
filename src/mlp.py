import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mlp_model.pt")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# ── Hyperparameters (adjust these) ──────────────────────────────────────────
HIDDEN_DIM = 128
DROPOUT = 0.2
EPOCHS = 30
LEARNING_RATE = 0.001
K_FOLDS = 5
SEED = 42
# ────────────────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
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
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)


def load_train_data():
    x = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv")).values
    y = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv")).squeeze().values
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def load_test_data():
    x = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv")).values
    y = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv")).squeeze().values
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, epochs=EPOCHS,
          lr=LEARNING_RATE, k_folds=K_FOLDS, seed=SEED):
    """Train the MLP with cross-validation and save the best model."""
    torch.manual_seed(seed)
    x, y = load_train_data()

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_results = []
    best_model = None
    best_f1 = 0.0
    train_f1_per_epoch = []
    val_f1_per_epoch = []

    fold_iter = tqdm(enumerate(skf.split(x, y)), total=k_folds, desc="Folds")
    for fold, (train_idx, val_idx) in fold_iter:
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = MLP(input_dim=x.shape[1], hidden_dim=hidden_dim, dropout=dropout)
        class_counts = torch.bincount(y_train)
        class_weights = 1.0 / class_counts.float()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_f1_history = []
        val_f1_history = []

        for epoch in tqdm(range(epochs), desc=f"Training Fold {fold + 1}"):
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
        val_pred = torch.argmax(val_preds, dim=1).numpy()
        cm = confusion_matrix(y_val.numpy(), val_pred)
        print(f"Confusion Matrix for Fold {fold + 1}:\n{cm}")
        fold_results.append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model

        train_f1_per_epoch.append(train_f1_history)
        val_f1_per_epoch.append(val_f1_history)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(best_model.state_dict(), MODEL_PATH)
    print(f"\nBest model saved to {MODEL_PATH}")

    # Plot learning curves
    avg_train_f1 = np.mean(train_f1_per_epoch, axis=0)
    avg_val_f1 = np.mean(val_f1_per_epoch, axis=0)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure()
    plt.plot(avg_train_f1, label="Train F1")
    plt.plot(avg_val_f1, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("MLP Cross-Validation Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "mlp_cv_learning_curve.png"))
    plt.close()

    print(f"Average Validation F1 across folds: {np.mean(fold_results):.4f}")
    return best_model


def test(hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
    """Load the saved model and evaluate on the test set."""
    x_test, y_test = load_test_data()

    model = MLP(input_dim=x_test.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    pbar = tqdm(total=3, desc="Testing MLP")

    pbar.set_postfix_str("Loading data & model")
    pbar.update(1)

    pbar.set_postfix_str("Running inference")
    with torch.no_grad():
        outputs = model(x_test)
        predictions = torch.argmax(outputs, dim=1).numpy()
    pbar.update(1)

    pbar.set_postfix_str("Evaluating")
    pbar.update(1)
    pbar.close()

    print("MLP Confusion Matrix:")
    print(confusion_matrix(y_test.numpy(), predictions))
    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), predictions,
                                target_names=["Distracting", "Focus-Friendly"]))


def plot(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, epochs=EPOCHS,
         lr=LEARNING_RATE, k_folds=K_FOLDS, seed=SEED):
    """Generate confusion matrices (train & test) and learning curves."""
    torch.manual_seed(seed)

    # Load model
    x_train, y_train = load_train_data()
    x_test, y_test = load_test_data()

    model = MLP(input_dim=x_train.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    with torch.no_grad():
        train_preds = torch.argmax(model(x_train), dim=1).numpy()
        test_preds = torch.argmax(model(x_test), dim=1).numpy()

    # Confusion matrix — training set
    cm_train = confusion_matrix(y_train.numpy(), train_preds)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                                        display_labels=["Distracting", "Focus-Friendly"])
    disp_train.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Training Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "mlp_confusion_matrix_train.png"))
    plt.close()

    # Confusion matrix — test set
    cm_test = confusion_matrix(y_test.numpy(), test_preds)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                                       display_labels=["Distracting", "Focus-Friendly"])
    disp_test.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "mlp_confusion_matrix_test.png"))
    plt.close()

    # Classification reports
    print("=== Training Set ===")
    print(classification_report(y_train.numpy(), train_preds,
                                target_names=["Distracting", "Focus-Friendly"]))
    print("=== Test Set ===")
    print(classification_report(y_test.numpy(), test_preds,
                                target_names=["Distracting", "Focus-Friendly"]))

    # Learning curves — retrain with increasing data sizes
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_f1_means = []
    train_f1_stds = []
    val_f1_means = []
    val_f1_stds = []
    actual_sizes = []

    for frac in tqdm(train_sizes, desc="Learning curve"):
        fold_train_f1s = []
        fold_val_f1s = []

        for train_idx, val_idx in skf.split(x_train, y_train):
            x_tr, y_tr = x_train[train_idx], y_train[train_idx]
            x_val, y_val = x_train[val_idx], y_train[val_idx]

            # Subset training data
            n = max(1, int(len(x_tr) * frac))
            x_tr, y_tr = x_tr[:n], y_tr[:n]

            m = MLP(input_dim=x_tr.shape[1], hidden_dim=hidden_dim, dropout=dropout)
            class_counts = torch.bincount(y_tr)
            class_weights = 1.0 / class_counts.float()
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(m.parameters(), lr=lr)

            for _ in range(epochs):
                m.train()
                optimizer.zero_grad()
                loss = criterion(m(x_tr), y_tr)
                loss.backward()
                optimizer.step()

            m.eval()
            with torch.no_grad():
                tr_f1 = f1_score(y_tr, torch.argmax(m(x_tr), dim=1), average="weighted")
                vl_f1 = f1_score(y_val, torch.argmax(m(x_val), dim=1), average="weighted")
            fold_train_f1s.append(tr_f1)
            fold_val_f1s.append(vl_f1)

        actual_sizes.append(n)
        train_f1_means.append(np.mean(fold_train_f1s))
        train_f1_stds.append(np.std(fold_train_f1s))
        val_f1_means.append(np.mean(fold_val_f1s))
        val_f1_stds.append(np.std(fold_val_f1s))

    actual_sizes = np.array(actual_sizes)
    train_f1_means = np.array(train_f1_means)
    train_f1_stds = np.array(train_f1_stds)
    val_f1_means = np.array(val_f1_means)
    val_f1_stds = np.array(val_f1_stds)

    test_f1 = f1_score(y_test.numpy(), test_preds, average="weighted")

    plt.figure(figsize=(10, 6))
    plt.plot(actual_sizes, train_f1_means, label="Training score", marker="o")
    plt.fill_between(actual_sizes, train_f1_means - train_f1_stds,
                     train_f1_means + train_f1_stds, alpha=0.2)
    plt.plot(actual_sizes, val_f1_means, label="Validation score", marker="s")
    plt.fill_between(actual_sizes, val_f1_means - val_f1_stds,
                     val_f1_means + val_f1_stds, alpha=0.2)
    plt.axhline(y=test_f1, color="r", linestyle="--", label=f"Test score ({test_f1:.3f})")
    plt.title("Learning Curve: MLP")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score (weighted)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "mlp_learning_curve.png"))
    plt.close()

    print("All plots saved to", FIGURES_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLP train / test / plot")
    parser.add_argument("action", choices=["train", "test", "plot"],
                        help="Action to perform")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--folds", type=int, default=K_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.action == "train":
        train(hidden_dim=args.hidden_dim, dropout=args.dropout,
              epochs=args.epochs, lr=args.lr, k_folds=args.folds,
              seed=args.seed)
    elif args.action == "test":
        test(hidden_dim=args.hidden_dim, dropout=args.dropout)
    elif args.action == "plot":
        plot(hidden_dim=args.hidden_dim, dropout=args.dropout,
             epochs=args.epochs, lr=args.lr, k_folds=args.folds,
             seed=args.seed)
