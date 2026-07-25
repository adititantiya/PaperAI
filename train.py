import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv("data/historical_data.csv")

# Features (inputs)
X = df[
    ["stock_flow","filler_flow","steam_pressure","machine_speed",
     "moisture","ash","basis_weight","target_basis_weight","bw_deviation"]]

# Target (output)
y = df["off_spec"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
joblib.dump(accuracy, "models/accuracy.pkl")
print(f"\nAccuracy: {accuracy*100:.2f}%\n")

print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "models/random_forest.pkl")

print("\n✅ Model saved successfully!")

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8,5))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()

plt.savefig("models/feature_importance.png")

print("✅ Feature Importance graph saved!")

# Save Confusion Matrix
plt.figure(figsize=(5,5))

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")

print("✅ Confusion Matrix saved!")