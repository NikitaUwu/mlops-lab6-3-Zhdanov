import pandas as pd
import great_expectations as ge
from datetime import datetime


def validate_data():
    """Валидация данных перед обучением"""
    # Загрузка данных
    data_path = "data/raw/dataset.csv"
    df = pd.read_csv(data_path)

    # Создание валидатора
    context = ge.get_context()
    validator = context.get_validator(
        batch_request={
            "path": data_path,
            "datasource_name": "pandas_datasource",
            "data_connector_name": "default_inferred_data_connector_name",
            "data_asset_name": "dataset"
        },
        expectation_suite_name="data_validation_suite"
    )

    # Добавление ожиданий
    validator.expect_column_values_to_not_be_null("customer_id")
    validator.expect_column_values_to_be_between("age", min_value=18, max_value=100)
    validator.expect_column_values_to_be_in_set("gender", value_set=["M", "F"])
    validator.expect_column_mean_to_be_between("monthly_charges", min_value=0, max_value=200)

    # Запуск валидации
    results = validator.validate()

    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"reports/data_validation_{timestamp}.html"
    validator.save_expectation_suite(discard_failed_expectations=False)
    validator.build_gallery()

    if not results["success"]:
        print("Data validation failed!")
        for result in results["results"]:
            if not result["success"]:
                print(f"- {result['expectation_config']['expectation_type']}: {result['result']['element_count']} records failed")
        return False

    print("Data validation passed!")
    return True


if __name__ == "__main__":
    validate_data()