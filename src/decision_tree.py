import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data")
MERGED_CSV = os.path.join(BASE_DIR, "data", "merged_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "decision_tree_model.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

METADATA_COLS = ["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std", "label"]

# ── Hyperparameters (adjust these) ──────────────────────────────────────────
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 5
CLASS_WEIGHT = "balanced"
K_FOLDS = 5
SEED = 42
# ────────────────────────────────────────────────────────────────────────────


def load_train_data():
    x = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv")).squeeze()
    return x, y


def load_test_data():
    x = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv")).squeeze()
    return x, y


def get_feature_names():
    merged_df = pd.read_csv(MERGED_CSV)
    return [col for col in merged_df.columns if col not in METADATA_COLS]


def build_classifier(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
                     min_samples_leaf=MIN_SAMPLES_LEAF,
                     class_weight=CLASS_WEIGHT, seed=SEED):
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=seed,
    )


def train(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
          min_samples_leaf=MIN_SAMPLES_LEAF, class_weight=CLASS_WEIGHT,
          k_folds=K_FOLDS, seed=SEED):
    """Train a decision tree with cross-validation and save the model."""
    x, y = load_train_data()

    clf = build_classifier(max_depth, min_samples_split, min_samples_leaf,
                           class_weight, seed)
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    f1_scores = []
    for fold, (train_idx, val_idx) in tqdm(enumerate(cv.split(x, y)),
                                           total=k_folds, desc="CV Folds"):
        fold_clf = build_classifier(max_depth, min_samples_split,
                                    min_samples_leaf, class_weight, seed)
        fold_clf.fit(x.iloc[train_idx], y.iloc[train_idx])
        preds = fold_clf.predict(x.iloc[val_idx])
        f1 = f1_score(y.iloc[val_idx], preds, average="weighted")
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    print("F1 Weighted Scores Per Fold:", f1_scores)
    print("Mean F1 Weighted Score:", np.mean(f1_scores))

    print("Fitting on full training set...")
    clf.fit(x, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": clf,
        "hparams": {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "k_folds": k_folds,
            "seed": seed,
        },
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return clf


def load_model():
    """Load saved checkpoint and return the classifier and hyperparameters."""
    checkpoint = joblib.load(MODEL_PATH)

    # Support both old (raw clf) and new (dict with hparams) formats
    if isinstance(checkpoint, dict) and "hparams" in checkpoint:
        return checkpoint["model"], checkpoint["hparams"]

    return checkpoint, {
        "max_depth": MAX_DEPTH, "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF, "class_weight": CLASS_WEIGHT,
        "k_folds": K_FOLDS, "seed": SEED,
    }


def test():
    """Load the saved model and evaluate on the test set."""
    pbar = tqdm(total=3, desc="Testing Decision Tree")

    pbar.set_postfix_str("Loading data & model")
    x_test, y_test = load_test_data()
    clf, hparams = load_model()
    print(f"Loaded model with hparams: {hparams}")
    pbar.update(1)

    pbar.set_postfix_str("Predicting")
    y_pred = clf.predict(x_test)
    pbar.update(1)

    pbar.set_postfix_str("Evaluating")
    pbar.update(1)
    pbar.close()

    print("Decision Tree Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Distracting", "Focus-Friendly"]))


def plot():
    """Generate all decision tree visualizations."""
    clf, hparams = load_model()
    max_depth = hparams.get("max_depth", MAX_DEPTH)
    min_samples_split = hparams.get("min_samples_split", MIN_SAMPLES_SPLIT)
    min_samples_leaf = hparams.get("min_samples_leaf", MIN_SAMPLES_LEAF)
    class_weight = hparams.get("class_weight", CLASS_WEIGHT)
    k_folds = hparams.get("k_folds", K_FOLDS)
    seed = hparams.get("seed", SEED)
    print(f"Plotting with saved hparams: {hparams}")

    x, y = load_train_data()
    feature_names = get_feature_names()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Tree structure
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names,
              class_names=["Distracting", "Focus-Friendly"],
              filled=True, fontsize=10)
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "dt_tree.png"))
    plt.close()

    # Top 10 feature importances
    importances = clf.feature_importances_
    indices = importances.argsort()[::-1]
    top_n = 10
    top_features = [feature_names[i] for i in indices[:top_n]]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices[:top_n]][::-1])
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "dt_feature_importances.png"))
    plt.close()

    # Confusion matrix — training set
    y_pred_train = clf.predict(x)
    cm_train = confusion_matrix(y, y_pred_train)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                                        display_labels=["Distracting", "Focus-Friendly"])
    disp_train.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Training Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "dt_confusion_matrix_train.png"))
    plt.close()

    # Confusion matrix — test set
    x_test, y_test = load_test_data()
    y_pred_test = clf.predict(x_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                                       display_labels=["Distracting", "Focus-Friendly"])
    disp_test.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "dt_confusion_matrix_test.png"))
    plt.close()

    # Classification reports
    print("=== Training Set ===")
    print(classification_report(y, y_pred_train,
                                target_names=["Distracting", "Focus-Friendly"]))
    print("=== Test Set ===")
    print(classification_report(y_test, y_pred_test,
                                target_names=["Distracting", "Focus-Friendly"]))

    # Learning curves
    fresh_clf = build_classifier(max_depth, min_samples_split,
                                 min_samples_leaf, class_weight, seed)
    train_sizes, train_scores, val_scores = learning_curve(
        fresh_clf, x, y, cv=k_folds, scoring="f1_weighted",
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    test_f1 = f1_score(y_test, y_pred_test, average="weighted")

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training score", marker="o")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, label="Validation score", marker="s")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.axhline(y=test_f1, color="r", linestyle="--", label=f"Test score ({test_f1:.3f})")
    plt.title("Learning Curve: Decision Tree")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score (weighted)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "dt_learning_curve.png"))
    plt.close()

    print("All plots saved to", FIGURES_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decision Tree train / test / plot")
    parser.add_argument("action", choices=["train", "test", "plot"],
                        help="Action to perform")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--min-samples-split", type=int, default=MIN_SAMPLES_SPLIT)
    parser.add_argument("--min-samples-leaf", type=int, default=MIN_SAMPLES_LEAF)
    parser.add_argument("--folds", type=int, default=K_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.action == "train":
        train(max_depth=args.max_depth,
              min_samples_split=args.min_samples_split,
              min_samples_leaf=args.min_samples_leaf,
              k_folds=args.folds, seed=args.seed)
    elif args.action == "test":
        test()
    elif args.action == "plot":
        plot()
