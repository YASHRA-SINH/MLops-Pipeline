import yaml
import pandas as pd
import os
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

data = load_wine()
X, y = data.data, data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

os.makedirs("data", exist_ok=True)
df.to_csv(params["dataset"], index=False)

if params["preprocessing"]["scale"]:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["training"]["test_size"],
    random_state=params["training"]["random_state"]
)

os.makedirs("data/processed", exist_ok=True)
pd.DataFrame(X_train).assign(target=y_train).to_csv("data/processed/train.csv", index=False)
pd.DataFrame(X_test).assign(target=y_test).to_csv("data/processed/test.csv", index=False)

print("Preprocessing complete. Processed files saved in data/processed/")
