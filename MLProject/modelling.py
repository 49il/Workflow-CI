import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# KOREKSI: Set tracking URI agar mlruns selalu berada di direktori aktif
mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")

# Load dataset
df = pd.read_csv("heart_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy"
)

# Environment untuk model artifact
custom_env = {
    "name": "heart-env",
    "channels": ["conda-forge", "nodefaults"],
    "dependencies": [
        "python=3.10",
        "pandas",
        "scikit-learn=1.5.2",
        "matplotlib",
        "pip",
        {"pip": ["mlflow==2.19.0"]}
    ]
}

with mlflow.start_run():
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Logging Metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    
    # Logging Model dengan environment aman
    mlflow.sklearn.log_model(best_model, "random_forest_model", conda_env=custom_env)

    # Logging Artifacts
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")