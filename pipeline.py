import pandas as pd
import os
import numpy as np
import shutil


def _merge(
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


def _count(input_path: str, output_path: str, groupby: str, output_colname: str):
    """
    Counts the number of occurences in a specific target column.
    Can be used to, e.g., count the number of diagnoses, medications, or procedures for a patient/visit.
    """
    df = pd.read_parquet(input_path)

    df = df.groupby(groupby).count().iloc[:, 0]
    df.name = output_colname
    df = df.reset_index(drop=False)

    df.to_parquet(output_path)


def _find_unique_ids(
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

    # load input
    df = pd.read_parquet(input_path)

    # match
    df[target_col] = df[target_col].astype(str)
    if match_on_first > 0:
        df[target_col] = df[target_col].str[:match_on_first]
    unq_id = df[df[target_col].isin(match_on)]

    # select unique identifiers which match the strings
    unq_id = unq_id[id_cols]
    unq_id = unq_id.groupby(id_cols).first().reset_index(drop=False)

    # write output
    unq_id.to_parquet(output_path)


def _ingest_data(
    input_path: str,
    output_path: str,
    usecols: list = [],
):
    """
    Reads data and writes to shared volume mount in parquet.
    Currently only support csv and gzip compression
    """
    if len(usecols) == 0:
        usecols = None
    split = input_path.split(".")
    if "gz" in split:
        compression = "gzip"
    else:
        compression = None
    if "csv" in split:
        df = pd.read_csv(input_path, compression=compression, usecols=usecols)
    elif "txt" in split:
        df = pd.read_csv(
            input_path, compression=compression, usecols=usecols, delimiter="\t"
        )

    df.to_parquet(output_path)
