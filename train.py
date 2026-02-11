

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("attention_relapse_dataset.csv")

# REMOVE leakage feature
X = df.drop(columns=[
    "session_id",
    "attention_label",
    "degradation_score"   # CRITICAL FIX
])

y = df["attention_label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "attention_model.pkl")

print("Model trained and saved successfully.")
