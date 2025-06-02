import os
import pandas as pd


def merge(
    input_path: str,
    output_path: str,
    input_path_merge: str,
    merge_on: list,
    how: str = "left",
):
    """
    Merges two dataframes together based on one or more columns.
    The first input path is the "base" dataframe on which the second input path is merged.
    """
    df = pd.read_parquet(input_path)
    df_merge = pd.read_parquet(input_path_merge)
    df = df.merge(df_merge, how=how, on=merge_on)

    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    input_path_merge = os.environ.get("input_path_merge")
    merge_on = os.environ.get("merge_on", "").split(",")
    how = os.environ.get("how", "left")

    merge(
        input_path=input_path,
        output_path=output_path,
        input_path_merge=input_path_merge,
        merge_on=merge_on,
        how=how
    ) 