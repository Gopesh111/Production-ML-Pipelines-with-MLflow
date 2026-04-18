import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("comparison-experiment")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params_list = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
]

for params in params_list:
    with mlflow.start_run():

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        print(f"Params: {params}, Accuracy: {acc}")