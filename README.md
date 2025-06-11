# Classifying Emotionality of Music

A machine learning project for predicting whether a song is focus-friendly or distracting based on audio features and emotion ratings.

## Overview

This project uses machine learning models to classify songs as either focus-friendly or distracting based on their acoustic and emotional features. It leverages the DEAM dataset, which provides per-song valence and arousal annotations and acoustic features. Two models were implemented and evaluated: a decision tree for interpretability and a multi-layer perceptron (MLP) for predictive performance.

## Project Structure

```
Classifying-Emotionality-of-Music/
├── data/
│   ├── annotations/              # Static valence/arousal annotations
│   ├── features/                 # openSMILE feature CSVs for each song
│   ├── merged_features.csv       # Combined features + labels per song
│   └── processed_data/           # Scaled train/test splits for modeling
├── models/                       # Saved model weights (MLP, decision tree)
├── plots/                        # Learning curves and confusion matrices
├── src/                          # Main Python scripts
│   ├── merge_features.py         # Merges annotations and features into a dataset
│   ├── split_data.py             # Splits merged data into train/test sets
│   ├── train_dt.py               # Trains and evaluates the decision tree model
│   ├── test_dt.py                # Tests the decision tree on held-out data
│   ├── mlp_train.py              # Trains MLP using cross-validation and tuning
│   ├── test_mlp.py               # Tests the trained MLP on held-out data
│   ├── plot_features.py          # Plots feature importance and learning curves
├── README.md                     # This file
```

## Key Files

- `merge_features.py`:  
  Combines per-song openSMILE feature CSVs with valence/arousal annotations into a single labeled dataset.

- `split_data.py`:  
  Splits the merged dataset into train/test sets and scales features using `StandardScaler`.

- `train_dt.py`:  
  Trains a decision tree classifier with cross-validation and optional grid search. Saves trained model and plots feature importances.

- `test_dt.py`:  
  Loads a trained decision tree and evaluates it on the test set.

- `mlp_train.py`:  
  Trains a multi-layer perceptron with support for 1–3 hidden layers, dropout tuning, and class weighting. Performs 5-fold cross-validation and logs performance.

- `test_mlp.py`:  
  Loads the best saved MLP model and evaluates it on the test set with a classification report and a confusion matrix.

- `plot_features.py`:  
  Generates learning curves and visual summaries of model performance and feature contributions.

## Results Summary

| Model          | Accuracy | F1 (Focus-Friendly) | F1 (Macro) |
|----------------|----------|---------------------|------------|
| Decision Tree  | 68%      | 0.30                | 0.55       |
| MLP (3-layer)  | 67%      | 0.31                | 0.55       |

## Citation
Developed by Alea Minar as a course project for CS472 Machine Learning at the University of Oregon.

This project uses the DEAM dataset (Database for Emotional Analysis of Music):
Eyben, F., Weninger, F., Gross, F., and Schuller, B. (2013). Recent developments in openSMILE, the Munich open-source multimedia feature extractor. In Proceedings of the 21st ACM International Conference on Multimedia (MM ’13), pages 835–838, Barcelona, Spain. ACM.
