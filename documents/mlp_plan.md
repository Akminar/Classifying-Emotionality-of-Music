# MLP Training & Testing Plan

## Step 1: Data Preparation

Run the data pipeline (skip if `processed_data/` already exists):

```bash
cd src
python data_processor.py split
```

## Step 2: Baseline Training

Train with the current default hyperparameters to establish a baseline:

```bash
python mlp.py train
```

| Hyperparameter | Default Value |
|---|---|
| `hidden_dim` | 128 |
| `dropout` | 0.2 |
| `epochs` | 30 |
| `lr` | 0.001 |
| `k_folds` | 5 |
| `seed` | 42 |

Record the **mean F1 weighted score** from 5-fold CV.

## Step 3: Hyperparameter Tuning

Run these experiments one at a time, changing one parameter from baseline:

### A. Hidden dimension (controls model capacity)

```bash
python mlp.py train --hidden-dim 32
python mlp.py train --hidden-dim 64
python mlp.py train --hidden-dim 256
python mlp.py train --hidden-dim 512
```

### B. Dropout (controls regularization)

```bash
python mlp.py train --dropout 0.0
python mlp.py train --dropout 0.1
python mlp.py train --dropout 0.3
python mlp.py train --dropout 0.5
```

### C. Learning rate (controls step size)

```bash
python mlp.py train --lr 0.0001
python mlp.py train --lr 0.0005
python mlp.py train --lr 0.005
python mlp.py train --lr 0.01
```

### D. Epochs (controls training duration)

```bash
python mlp.py train --epochs 10
python mlp.py train --epochs 50
python mlp.py train --epochs 100
```

Record CV F1 for each run. Pick the best value for each parameter, then combine them in a final training run.

## Step 4: Final Training

Train with the best combination found in Step 3:

```bash
python mlp.py train --hidden-dim <best> --dropout <best> --lr <best> --epochs <best>
```

This saves the model to `models/mlp_model.pt`.

## Step 5: Test Evaluation

Evaluate on the held-out test set (only do this once, with your final model):

```bash
python mlp.py test
```

Review the confusion matrix and classification report (precision, recall, F1 per class).

## Step 6: Generate Visualizations

```bash
python mlp.py plot
```

This saves to `figures/`:

- `mlp_confusion_matrix_train.png` — confusion matrix on training set
- `mlp_confusion_matrix_test.png` — confusion matrix on test set
- `mlp_learning_curve.png` — train, validation, and test F1 across dataset sizes

Note: the learning curve retrains the MLP at 10 data sizes across 5 folds, so this step takes longer than training.

## Results Tracking

| Run | hidden_dim | dropout | lr | epochs | Mean CV F1 |
|---|---|---|---|---|---|
| Baseline | 128 | 0.2 | 0.001 | 30 | 0.6472 |
| A1 | 32 | 0.2 | 0.001 | 30 | 0.6504 |
| A2 | 64 | 0.2 | 0.001 | 30 | 0.6811 |
| A3 | 256 | 0.2 | 0.001 | 30 | 0.6531 |
| A4 | 512 | 0.2 | 0.001 | 30 | 0.6402 |
| B1 | 128 | 0.0 | 0.001 | 30 | 0.6542 |
| B2 | 128 | 0.1 | 0.001 | 30 | 0.6584 |
| B3 | 128 | 0.3 | 0.001 | 30 | 0.6399 |
| B4 | 128 | 0.5 | 0.001 | 30 | 0.6235 |
| C1 | 128 | 0.2 | 0.0001 | 30 | 0.4647 |
| C2 | 128 | 0.2 | 0.0005 | 30 | 0.6696 |
| C3 | 128 | 0.2 | 0.005 | 30 | 0.6974 |
| C4 | 128 | 0.2 | 0.01 | 30 | 0.7121 |
| D1 | 128 | 0.2 | 0.001 | 10 | 0.6661 |
| D2 | 128 | 0.2 | 0.001 | 50 | 0.6384 |
| D3 | 128 | 0.2 | 0.001 | 100 | 0.7154 |
| **Final** | 64 | 0.1 | 0.001 | 100 | 0.7544 |
| **Test** | | | | | |
