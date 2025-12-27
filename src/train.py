import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Начало эксперимента
mlflow.start_run()
# Загрузка данных

data_path = "data/processed/train.csv"
df = pd.read_csv(data_path)

# Разделение на признаки и целевую переменную
X = df.drop("churn", axis=1)
y = df["churn"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Логирование параметров
n_estimators = 100
max_depth = 10
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)

# Создание пайплайна
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    ))
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Оценка качества
y_pred = pipeline.predict(X_test)
f1 = f1_score(y_test, y_pred)
mlflow.log_metric("f1_score", f1)
print(f"F1 Score: {f1:.4f}")

mlflow.sklearn.log_model(pipeline, "model")
# Регистрация модели в Model Registry

model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
model_details = mlflow.register_model(model_uri, "churn-prediction")

# Завершение эксперимента
mlflow.end_run()