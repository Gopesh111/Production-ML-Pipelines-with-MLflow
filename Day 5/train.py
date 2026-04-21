import os
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
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
mlflow.set_experiment("day5-hyperparam-exp")

# =========================
# DATA (make it realistic)
# =========================
X, y = load_iris(return_X_y=True)
X = X + np.random.normal(0, 0.6, X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4
)

# =========================
# OBJECTIVE FUNCTION
# =========================
def objective(trial):
    with mlflow.start_run(nested=True):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }

        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Trial {trial.number}")

        os.makedirs("artifacts", exist_ok=True)
        path = f"artifacts/cm_trial_{trial.number}.png"
        plt.savefig(path)
        plt.close()

        mlflow.log_artifact(path)

        return acc


# =========================
# OPTUNA STUDY
# =========================
study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=20)

print("Best Trial:")
print(study.best_trial.params)