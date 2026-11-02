
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Remove unnecessary columns and failure-type labels
columns_to_drop = [
    "UDI", "Product ID", "TWF", "HDF",
    "PWF", "OSF", "RNF"
]

df = df.drop(columns=columns_to_drop)

# Separate features and target
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Convert machine type into numerical values
X = pd.get_dummies(X, columns=["Type"], drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/failure_model.pkl")

# Save feature column names for the dashboard
joblib.dump(list(X.columns), "models/features.pkl")

print("\nModel saved successfully!")
