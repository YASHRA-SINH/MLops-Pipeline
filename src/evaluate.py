import yaml
import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load train set
df = pd.read_csv("data/processed/train.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Define model
model = DecisionTreeClassifier(**params["model"]["params"])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=params["training"]["random_state"])
scores = cross_validate(
    model, X, y, cv=cv,
    scoring=["accuracy", "f1_macro"],
    return_train_score=False
)

# ✅ Metrics.json for DVC (dict, not list)
metrics = {
    "accuracy": float(scores["test_accuracy"].mean()),
    "f1": float(scores["test_f1_macro"].mean())
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# (Optional) keep fold-wise results for inspection
fold_metrics = [
    {"fold": i+1, "accuracy": float(acc), "f1": float(f1)}
    for i, (acc, f1) in enumerate(zip(scores["test_accuracy"], scores["test_f1_macro"]))
]
with open("fold_metrics.json", "w") as f:
    json.dump(fold_metrics, f, indent=4)

# Also save results.json
os.makedirs("experiments", exist_ok=True)
with open("experiments/results.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation complete ✅", metrics)
