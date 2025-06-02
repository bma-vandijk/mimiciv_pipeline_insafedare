import os
import pandas as pd


def feature_selection(
    input_path: str,
    output_path: str,
    columns: list,
):
    """
    Manually select a set of features to retain.
    """
    df = pd.read_parquet(input_path)
    df = df[columns]
    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    columns = os.environ.get("columns", "").split(",")

    feature_selection(
        input_path=input_path,
        output_path=output_path,
        columns=columns
    ) 