import os
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# =========================
# CONFIG
# =========================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("day4-centralized-exp")

# =========================
# DATA (make it realistic)
# =========================
X, y = load_iris(return_X_y=True)

# Add noise (avoid perfect results)
X = X + np.random.normal(0, 0.6, X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4
)

# =========================
# PARAM GRID
# =========================
param_grid = [
    {"n_estimators": 10, "max_depth": 2},
    {"n_estimators": 50, "max_depth": 4},
    {"n_estimators": 100, "max_depth": None},
]

# =========================
# TRAIN LOOP
# =========================
for params in param_grid:
    with mlflow.start_run():

        # Log params
        mlflow.log_params(params)

        # Train
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # =========================
        # CONFUSION MATRIX
        # =========================
        cm = confusion_matrix(y_test, preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Params: {params}")

        os.makedirs("artifacts", exist_ok=True)
        file_path = f"artifacts/cm_{params['n_estimators']}_{params['max_depth']}.png"

        plt.savefig(file_path)
        plt.close()

        mlflow.log_artifact(file_path)

        print(f"Run: {params} | Acc: {acc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")