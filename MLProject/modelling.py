import pandas as pd
import mlflow
import mlflow.sklearn
import os
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Inisialisasi DagsHub sesuai link Anda
dagshub.init(repo_owner='49il', repo_name='heart-mlflow', mlflow=True)

# Set tracking URI
mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")

# 1. Load dataset (Pastikan file ini ada di folder MLProject)
df = pd.read_csv("heart_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Hyperparameter Tuning
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

# 3. Definisi Environment (Python 3.12.7 sesuai instruksi)
custom_env = {
    "name": "heart-env",
    "channels": ["conda-forge", "nodefaults"],
    "dependencies": [
        "python=3.12.7",
        "pandas",
        "scikit-learn=1.5.2",
        "matplotlib",
        "pip",
        {"pip": [
            "mlflow==2.19.0",
            "setuptools",
            "wheel"
        ]}
    ]
}

# 4. Eksekusi Training & Logging
with mlflow.start_run():
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_params(grid_search.best_params_)

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=best_model, 
        artifact_path="random_forest_model", 
        conda_env=custom_env
    )

    # Log Artifacts (Syarat Advanced)
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Random Forest Accuracy: {accuracy:.4f}")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

print("Training selesai. Silakan cek DagsHub untuk melihat hasilnya.")