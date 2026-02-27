# Decision Tree Training & Testing Plan

## Step 1: Data Preparation

Run the data pipeline (skip if `processed_data/` already exists):

```bash
cd src
python data_processor.py split
```

This loads `merged_features.csv`, scales features with `StandardScaler`, and creates an 80/20 stratified train/test split.

## Step 2: Baseline Training

Train with the current default hyperparameters to establish a baseline:

```bash
python decision_tree.py train
```

| Hyperparameter | Default Value |
|---|---|
| `max_depth` | 10 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 5 |
| `class_weight` | balanced |
| `k_folds` | 5 |
| `seed` | 42 |

Record the **mean F1 weighted score** from 5-fold CV.

## Step 3: Hyperparameter Tuning

Run these experiments one at a time, changing one parameter from baseline:

### A. Tree depth (controls model complexity)

```bash
python decision_tree.py train --max-depth 3
python decision_tree.py train --max-depth 5
python decision_tree.py train --max-depth 15
python decision_tree.py train --max-depth 20
```

### B. Minimum samples per leaf (controls overfitting)

```bash
python decision_tree.py train --min-samples-leaf 1
python decision_tree.py train --min-samples-leaf 10
python decision_tree.py train --min-samples-leaf 20
```

### C. Minimum samples to split (controls when nodes stop splitting)

```bash
python decision_tree.py train --min-samples-split 5
python decision_tree.py train --min-samples-split 10
python decision_tree.py train --min-samples-split 20
```

Record CV F1 for each run. Pick the best value for each parameter, then combine them in a final training run.

## Step 4: Final Training

Train with the best combination found in Step 3:

```bash
python decision_tree.py train --max-depth <best> --min-samples-leaf <best> --min-samples-split <best>
```

This saves the model to `models/decision_tree_model.pkl`.

## Step 5: Test Evaluation

Evaluate on the held-out test set (only do this once, with your final model):

```bash
python decision_tree.py test
```

Review the confusion matrix and classification report (precision, recall, F1 per class).

## Step 6: Generate Visualizations

```bash
python decision_tree.py plot
```

This saves to `figures/`:

- `dt_tree.png` — full tree structure
- `dt_feature_importances.png` — top 10 features
- `dt_confusion_matrix.png` — confusion matrix heatmap
- `dt_learning_curve.png` — train vs validation F1 across dataset sizes

## Results Tracking

| Run | max_depth | min_samples_split | min_samples_leaf | Mean CV F1 |
|---|---|---|---|---|
| Baseline | 10 | 2 | 5 | |
| A1 | 3 | 2 | 5 | |
| A2 | 5 | 2 | 5 | |
| A3 | 15 | 2 | 5 | |
| A4 | 20 | 2 | 5 | |
| B1 | 10 | 2 | 1 | |
| B2 | 10 | 2 | 10 | |
| B3 | 10 | 2 | 20 | |
| C1 | 10 | 5 | 5 | |
| C2 | 10 | 10 | 5 | |
| C3 | 10 | 20 | 5 | |
| **Final** | | | | |
| **Test** | | | | |
