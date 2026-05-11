import pandas as pd
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from mlflow.models.signature import infer_signature

# MATIKAN DAGSHUB UNTUK GITHUB ACTIONS AGAR TIDAK ERROR JSONDecodeError
# import dagshub
# dagshub.init(repo_owner="49il", repo_name="heart-mlflow", mlflow=True)

# Simpan artefak di folder lokal untuk Docker Build
mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")

# Kunci Environment (Penting untuk kestabilan CI/CD)
custom_env = {
    "name": "heart-env",
    "channels": ["conda-forge", "nodefaults"],
    "dependencies": [
        "python=3.12.7", 
        "pandas",
        "scikit-learn==1.6.1", 
        "matplotlib",
        "pip",
        {"pip": [
            "mlflow==2.19.0",
            "setuptools",
            "wheel"
        ]}
    ]
}

df = pd.read_csv("heart_preprocessing.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy")

with mlflow.start_run():
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(X_train, best_model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="random_forest_model",
        conda_env=custom_env,
        signature=signature
    )

    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Random Forest Accuracy: {accuracy:.4f}")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")