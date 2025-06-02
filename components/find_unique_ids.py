import os
import pandas as pd


def find_unique_ids(
    input_path: str,
    output_path: str,
    match_on: list,
    target_col: str,
    id_cols: list,
    match_on_first: int = 0,
):
    """
    Finds unique ids which match on some specified strings or values in a certain column.
    Can be used for cohort selection on string codes for, e.g., diseases, procedures, medications.
    """
    df = pd.read_parquet(input_path)

    df[target_col] = df[target_col].astype(str)
    if match_on_first > 0:
        df[target_col] = df[target_col].str[:match_on_first]
    unq_id = df[df[target_col].isin(match_on)]

    unq_id = unq_id[id_cols]
    unq_id = unq_id.groupby(id_cols).first().reset_index(drop=False)

    unq_id.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    match_on = os.environ.get("match_on", "").split(",")
    target_col = os.environ.get("target_col")
    id_cols = os.environ.get("id_cols", "").split(",")
    match_on_first = int(os.environ.get("match_on_first", "0"))

    find_unique_ids(
        input_path=input_path,
        output_path=output_path,
        match_on=match_on,
        target_col=target_col,
        id_cols=id_cols,
        match_on_first=match_on_first
    ) 