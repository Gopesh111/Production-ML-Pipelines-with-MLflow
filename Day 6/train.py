import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# CONFIG
# =========================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("day6-mlflow-models")

# =========================
# DATA
# =========================
X, y = load_iris(return_X_y=True)
X = X + np.random.normal(0, 0.5, X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

# =========================
# TRAIN + LOG MODEL
# =========================
with mlflow.start_run() as run:

    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    run_id = run.info.run_id   # ✅ capture here

    print(f"Model trained | Accuracy: {acc:.3f}")

# =========================
# LOAD MODEL BACK
# =========================

model_uri = f"runs:/{run_id}/model"

loaded_model = mlflow.pyfunc.load_model(model_uri)

sample_input = X_test[:5]
predictions = loaded_model.predict(sample_input)

print("\nLoaded Model Predictions:")
print(predictions)

# =========================
# INFERENCE TEST
# =========================
sample_input = X_test[:5]
predictions = loaded_model.predict(sample_input)

print("\nLoaded Model Predictions:")
print(predictions)