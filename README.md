# Production-ML-Pipelines-with-MLflow

#Day 1

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
