import os
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    mlflow.set_experiment("customer-churn-prediction")

    data_path = "data/processed/train.csv"
    df = pd.read_csv(data_path)

    if "churn" not in df.columns:
        raise ValueError("Training data must contain 'churn' column")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_estimators = 100
    max_depth = 10

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ))
    ])

    os.makedirs("models", exist_ok=True)

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("f1", f1)
        mlflow.log_metric("f1_score", f1)
        print(f"F1 Score: {f1:.4f}")

        # 1) Сохраняем локальный артефакт для Docker
        local_model_path = os.path.join("models", "model.joblib")
        joblib.dump(pipeline, local_model_path)
        mlflow.log_artifact(local_model_path, artifact_path="exported")

        # 2) Логируем модель в MLflow Artifacts
        mlflow.sklearn.log_model(pipeline, "model")

        # 3) Регистрируем и переводим в Production
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        mv = mlflow.register_model(model_uri, "churn-prediction")

        client = MlflowClient()
        client.transition_model_version_stage(
            name="churn-prediction",
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )

        print(f"Registered churn-prediction v{mv.version} -> Production (run_id={run_id})")
        print(f"Exported local model -> {local_model_path}")


if __name__ == "__main__":
    main()
