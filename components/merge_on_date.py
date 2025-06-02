import os
import pandas as pd


def merge_on_date(
    input_path: str,
    output_path: str,
    input_path_merge: str,
    date_col: str,
    date_col_merge: str,
    merge_on: str = None,
    direction: str = "nearest",
):
    """
    Merge on closest date (options are "nearest", "backward", "forward").
    Useful for adding data to a cohort which was measured closest to a certain event, e.g., admission or death.
    Optionally merge on merge_on column first to, e.g., ensure measurements are matched for the correct patient or visit.
    """
    df = pd.read_parquet(input_path)
    df_merge = pd.read_parquet(input_path_merge)

    df[date_col] = pd.to_datetime(df[date_col])
    df_merge[date_col_merge] = pd.to_datetime(df_merge[date_col_merge])
    df = df.sort_values(date_col)
    df_merge = df_merge.sort_values(date_col_merge)

    df = pd.merge_asof(
        df,
        df_merge,
        by=merge_on,
        left_on=date_col,
        right_on=date_col_merge,
        direction=direction,
    )

    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    input_path_merge = os.environ.get("input_path_merge")
    date_col = os.environ.get("date_col")
    date_col_merge = os.environ.get("date_col_merge")
    merge_on = os.environ.get("merge_on")
    direction = os.environ.get("direction", "nearest")

    merge_on_date(
        input_path=input_path,
        output_path=output_path,
        input_path_merge=input_path_merge,
        date_col=date_col,
        date_col_merge=date_col_merge,
        merge_on=merge_on,
        direction=direction
    ) 