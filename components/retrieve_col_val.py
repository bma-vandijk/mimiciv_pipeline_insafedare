import os
import pandas as pd


def retrieve_col_val(
    input_path: str,
    output_path: str,
    retrieve_string: str,
    match_type: str = "startswith",
    name_col: str = "result_name",
    val_col: str = "result_value",
):
    """
    Change table layout to col:val instead of col1:col_names,col2:col_values.
    Can be used to retrieve specific measurements from tables which store various measurement types in a single column.
    """
    df = pd.read_parquet(input_path)

    if match_type == "startswith":
        df = df[df[name_col].str.startswith(retrieve_string)]
    elif match_type == "endswith":
        df = df[df[name_col].str.endswith(retrieve_string)]
    elif match_type == "full_match":
        df = df[df[name_col] == retrieve_string]
    else:
        raise Exception("Invalid match type.")

    df = df.rename({val_col: retrieve_string}, axis=1)
    df = df.drop(name_col, axis=1)

    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    retrieve_string = os.environ.get("retrieve_string")
    match_type = os.environ.get("match_type", "startswith")
    name_col = os.environ.get("name_col", "result_name")
    val_col = os.environ.get("val_col", "result_value")

    retrieve_col_val(
        input_path=input_path,
        output_path=output_path,
        retrieve_string=retrieve_string,
        match_type=match_type,
        name_col=name_col,
        val_col=val_col
    ) 