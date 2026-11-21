# Task 3 - Credit Card Fraud Detection

## Objective
Classify credit-card transactions as genuine (0) or fraudulent (1).

## Dataset required
Download the credit-card fraud dataset specified by your internship instructions and place the CSV in this folder with the exact name `creditcard.csv`. It must contain a target column named `Class`, where 0 means genuine and 1 means fraud.

## Method
- Separate features and target
- Stratified train/test split
- Standardize features using training data only
- Logistic Regression with `class_weight="balanced"` to account for class imbalance
- Evaluate using precision, recall, F1-score, confusion matrix, average precision, and ROC-AUC

## Important
Fraud datasets are highly imbalanced. Accuracy alone can be misleading, so review precision and recall for the fraud class. This is an educational model, not a real banking fraud-decision system.

## Run
```bash
pip install -r requirements.txt
python main.py
```
