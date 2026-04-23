# Production-ML-Pipelines-with-MLflow

# Day 1

## Problem
Tracking ML experiments manually is messy and non-reproducible.

## Solution
Used MLflow to log:
- Parameters (n_estimators, max_depth)
- Metrics (accuracy)
- Model artifacts

## Steps
1. Train model
2. Log params + metrics using MLflow
3. Visualize using MLflow UI

## Run Instructions
```bash
pip install -r requirements.txt
python train.py
mlflow ui
```

# Day 2

## Problem
Single metric (like accuracy) is not enough to understand model performance.  
Difficult to debug model errors without visual insights.

## Solution
Used MLflow to log:
- Multiple metrics (accuracy, precision)
- Artifacts (confusion matrix plot)
- Model for each run

## Steps
1. Train model
2. Log parameters and multiple metrics
3. Generate confusion matrix
4. Save and log it as an artifact
5. Compare runs in MLflow UI

## Run Instructions
```bash
pip install -r requirements.txt

# create required folders (only once)
mkdir mlflow_server
mkdir mlflow_server/artifacts

# start tracking server
mlflow server --backend-store-uri sqlite:///mlflow_server/mlflow.db --default-artifact-root ./mlflow_server/artifacts

# run training script
python train.py
```

## Output
- Compared multiple runs using accuracy and precision  
- Visualized model errors using confusion matrix  
- Stored artifacts and models for each experiment  


# Day 3

## Problem
After multiple experiments, it is difficult to manage which model to use.  
No clear way to track versions or decide which model goes to production.

## Solution
Used MLflow Model Registry to:
- Version models automatically
- Manage lifecycle (Staging → Production)
- Track best-performing model

## Steps
1. Train model and log metrics  
2. Register model in MLflow  
3. Create multiple model versions  
4. Promote model to Staging or Production  
5. Compare and select best version  

## Run Instructions
```bash
# make sure tracking server is running
mlflow server --backend-store-uri sqlite:///mlflow_server/mlflow.db --default-artifact-root ./mlflow_server/artifacts

# run training script (will register models)
python train.py
```

## Output
- Model versions created (v1, v2, v3)  
- Ability to promote models to Staging/Production  
- Clear tracking of which model is active

# Day 4

## Problem
Local MLflow tracking is not suitable for team collaboration and persistent storage.

## Solution
Configured MLflow with a database backend (SQLite) to:
- Store experiment metadata centrally
- Enable persistence across sessions
- Support scalable tracking

## Steps
1. Set up MLflow tracking server
2. Configure SQLite as backend store
3. Run training script
4. Verify persistence after server restart

## Run Instructions
```bash
# start tracking server
mlflow server --backend-store-uri sqlite:///mlflow_server/mlflow.db --default-artifact-root ./mlflow_server/artifacts

# run training
python train.py
```

## Output
- Experiments stored in database
- Data persists across restarts
- Centralized tracking system ready

# Day 5

## Problem
Manual hyperparameter tuning is inefficient and hard to track.

## Solution
Integrated Optuna with MLflow to:
- Automatically explore parameter space
- Log all trials
- Identify best configuration

## Steps
1. Define objective function
2. Use Optuna to run multiple trials
3. Log each trial in MLflow
4. Compare results

## Run Instructions
```bash
pip install -r requirements.txt
python train.py
```

## Output
- 20+ runs automatically generated
- Best hyperparameters identified
- Full experiment tracking in MLflow

- # Day 6

## Problem
Saving models in raw formats (pickle) is not portable or standardized.

## Solution
Used MLflow Models to:
- Save models with standard format
- Enable easy loading and inference
- Support multiple frameworks via flavors

## Steps
1. Train model
2. Save model using MLflow
3. Load model using MLflow API
4. Run inference

## Run Instructions
```bash
python train.py
```

## Output
- Model saved in MLflow format
- Model reloaded successfully
- Predictions generated from loaded model
