import os
import pandas as pd


def to_numeric(
    input_path: str,
    output_path: str,
    columns: list,
):
    """
    Coerces columns to numeric dtype.
    Sets as NaN if not possible.
    """
    df = pd.read_parquet(input_path)
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    columns = os.environ.get("columns", "").split(",")

    to_numeric(
        input_path=input_path,
        output_path=output_path,
        columns=columns
    ) 