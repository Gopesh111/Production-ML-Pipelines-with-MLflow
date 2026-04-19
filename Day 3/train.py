import os
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# =========================
# CONFIG
# =========================
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "iris-prod-exp"
REGISTERED_MODEL_NAME = "iris-classifier"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# LOAD + MODIFY DATA
# =========================
X, y = load_iris(return_X_y=True)

# 🔥 Add noise (to avoid perfect accuracy)
X = X + np.random.normal(0, 0.5, X.shape)

# 🔥 Bigger test split + randomness
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=None
)

# =========================
# PARAM GRID (REAL VARIATION)
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

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)

        # =========================
        # CONFUSION MATRIX
        # =========================
        cm = confusion_matrix(y_test, preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"CM: {params}")

        os.makedirs("artifacts", exist_ok=True)
        file_name = f"artifacts/cm_{params['n_estimators']}_{params['max_depth']}.png"

        plt.savefig(file_name)
        plt.close()

        mlflow.log_artifact(file_name)

        # =========================
        # REGISTER MODEL
        # =========================
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        print(f"Params: {params} | Accuracy: {acc:.4f} | Precision: {precision:.4f}")