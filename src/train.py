import yaml
import pandas as pd
import pickle
import os
from sklearn.tree import DecisionTreeClassifier

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/processed/train.csv")
X_train = df.drop("target", axis=1)
y_train = df["target"]

model = DecisionTreeClassifier(**params["model"]["params"])
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training complete. Model saved to models/model.pkl")
