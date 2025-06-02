import os
import pandas as pd


def count(
    input_path: str,
    output_path: str,
    groupby: str,
    output_colname: str,
):
    """
    Counts the number of occurences in a specific target column.
    Can be used to, e.g., count the number of diagnoses, medications, or procedures for a patient/visit.
    """
    df = pd.read_parquet(input_path)

    df = df.groupby(groupby).count().iloc[:, 0]
    df.name = output_colname
    df = df.reset_index(drop=False)

    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    groupby = os.environ.get("groupby")
    output_colname = os.environ.get("output_colname")

    count(
        input_path=input_path,
        output_path=output_path,
        groupby=groupby,
        output_colname=output_colname
    ) 