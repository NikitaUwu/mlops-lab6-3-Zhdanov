import sys
import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe

DATA_PATH = "data/raw/dataset.csv"


def _safe_failure_line(obj) -> str:
    # Максимально совместимо между версиями GX: пытаемся сериализовать, иначе str()
    try:
        if hasattr(obj, "to_json_dict"):
            return str(obj.to_json_dict())
    except Exception:
        pass
    return str(obj)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    context = gx.get_context()

    ds_name = "pandas_ds"
    try:
        data_source = context.data_sources.get(ds_name)
    except Exception:
        data_source = context.data_sources.add_pandas(name=ds_name)

    asset_name = "dataset_asset"
    try:
        data_asset = data_source.get_asset(asset_name)
    except Exception:
        data_asset = data_source.add_dataframe_asset(name=asset_name)

    batch_def_name = "whole_dataframe"
    try:
        batch_definition = data_asset.get_batch_definition(batch_def_name)
    except Exception:
        batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_def_name)

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="data_validation_suite")

    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=1, max_value=None))
    suite.add_expectation(gxe.ExpectTableColumnCountToBeBetween(min_value=2, max_value=None))

    if "churn" in df.columns:
        target_col = "churn"
    elif "target" in df.columns:
        target_col = "target"
    else:
        print("Data validation failed: missing target column ('churn' or 'target')", file=sys.stderr)
        sys.exit(1)

    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=target_col))

    if pd.api.types.is_numeric_dtype(df[target_col]) or pd.api.types.is_bool_dtype(df[target_col]):
        suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column=target_col, value_set=[0, 1]))

    for id_col in ("customer_id", "id"):
        if id_col in df.columns:
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=id_col))

    results = batch.validate(expect=suite)

    if not results.success:
        print("Data validation failed!", file=sys.stderr)
        # Печатаем все провалившиеся ожидания без обращения к нестабильным полям
        for r in getattr(results, "results", []):
            if not getattr(r, "success", True):
                print("- " + _safe_failure_line(r), file=sys.stderr)
        sys.exit(1)

    print("Data validation passed.")


if __name__ == "__main__":
    main()
