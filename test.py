
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("attention_model.pkl")

# Load dataset
df = pd.read_csv("attention_relapse_dataset.csv")

X = df.drop(columns=[
    "session_id",
    "attention_label",
    "degradation_score"
])

y = df["attention_label"]

# SAME split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Predict ONLY on test data
predictions = model.predict(X_test)

# Save predictions
output = X_test.copy()
output["actual_label"] = y_test
output["predicted_label"] = predictions

output.to_csv("test_predictions.csv", index=False)

print("Test predictions saved to test_predictions.csv")
