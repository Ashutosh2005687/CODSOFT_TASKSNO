import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("IRIS FLOWER CLASSIFICATION")
print("-------------------------")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names
)
predicted_class = model.predict(sample)[0]
print("\nSample flower prediction:", iris.target_names[predicted_class])
