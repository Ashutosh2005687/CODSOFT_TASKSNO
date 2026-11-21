from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path("Advertising.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Advertising.csv not found. Put the project dataset in this folder. "
        "Expected columns: TV, Radio, Newspaper, Sales."
    )

data = pd.read_csv(DATA_FILE)
data.columns = data.columns.str.strip()

required = {"TV", "Radio", "Newspaper", "Sales"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

data = data[list(required)].dropna()
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("SALES PREDICTION")
print("----------------")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.3f}")
print(f"Root Mean Squared Error: {mean_squared_error(y_test, predictions) ** 0.5:.3f}")
print(f"R-squared: {r2_score(y_test, predictions):.3f}")

sample = pd.DataFrame([[150, 25, 20]], columns=["TV", "Radio", "Newspaper"])
print(f"\nPredicted sales for sample advertising spend: {model.predict(sample)[0]:.2f}")
