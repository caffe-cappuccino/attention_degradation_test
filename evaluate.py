

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV

# ----------------------------------
# 1. Load Dataset
# ----------------------------------

df = pd.read_csv("attention_relapse_dataset.csv")

X = df.drop(columns=[
    "session_id",
    "attention_label",
    "degradation_score"
])

y = df["attention_label"]

# ----------------------------------
# 2. Train/Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------
# 3. Model Comparison
# ----------------------------------

models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

print("\nMODEL COMPARISON (Accuracy)")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Bar Chart Comparison
plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()

# ----------------------------------
# 4. Cross Validation
# ----------------------------------

rf = RandomForestClassifier(n_estimators=300, random_state=42)

cv_scores = cross_val_score(rf, X, y, cv=5)

print("\nCross Validation Accuracy (RF):", cv_scores.mean())

# ----------------------------------
# 5. Hyperparameter Tuning
# ----------------------------------

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 10, 20]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

best_model = grid.best_estimator_

# ----------------------------------
# 6. Final Evaluation
# ----------------------------------

y_pred = best_model.predict(X_test)

print("\nFinal Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------------
# 7. ROC Curve (Multi-class)
# ----------------------------------

y_test_bin = label_binarize(y_test, classes=best_model.classes_)
y_score = best_model.predict_proba(X_test)

plt.figure()

for i, class_name in enumerate(best_model.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# ----------------------------------
# 8. Precision-Recall Curves
# ----------------------------------

plt.figure()

for i, class_name in enumerate(best_model.classes_):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=class_name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.show()

# ----------------------------------
# 9. Learning Curve
# ----------------------------------

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X,
    y,
    cv=5
)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation Score")
plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend()
plt.show()

# ----------------------------------
# 10. Feature Importance
# ----------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("Feature Importance")
plt.show()

# ----------------------------------
# 11. Correlation Heatmap
# ----------------------------------

plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Save best model
joblib.dump(best_model, "final_attention_model.pkl")

print("\nAdvanced Evaluation Complete.")
