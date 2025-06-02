import os
import pandas as pd


def drop_nan(
    input_path: str,
    output_path: str,
):
    """
    Drop na values from a dataframe.
    """
    df = pd.read_parquet(input_path)
    df = df.dropna().reset_index(drop=True)
    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")

    drop_nan(
        input_path=input_path,
        output_path=output_path
    ) 