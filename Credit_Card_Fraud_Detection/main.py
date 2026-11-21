from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, average_precision_score,
    roc_auc_score
)

DATA_FILE = Path("creditcard.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "creditcard.csv not found. Put the downloaded dataset in this folder."
    )

data = pd.read_csv(DATA_FILE)
if "Class" not in data.columns:
    raise ValueError("Expected target column 'Class' was not found.")

X = data.drop(columns=["Class"])
y = data["Class"]

if y.nunique() != 2:
    raise ValueError("The Class column should contain genuine (0) and fraud (1) labels.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features using training data only to avoid data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

print("CREDIT CARD FRAUD DETECTION")
print("---------------------------")
print("Class distribution (0 = genuine, 1 = fraud):")
print(y.value_counts().sort_index())
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print(f"Average Precision: {average_precision_score(y_test, probabilities):.4f}")
if y_test.nunique() == 2:
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
