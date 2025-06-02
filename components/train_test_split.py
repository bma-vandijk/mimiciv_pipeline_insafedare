import os
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_data(
    input_path: str,
    output_path: str,
    output_path_test: str,
    train_size: float = 0.7,
    stratify: str = None,
    seed: int = 0,
):
    """
    Splits a dataframe into a train and test split.
    """
    df = pd.read_parquet(input_path)

    stratify_col = df[stratify] if stratify is not None else None
    df_tr, df_te = train_test_split(
        df, train_size=train_size, stratify=stratify_col, random_state=seed
    )

    df_tr.to_parquet(output_path)
    df_te.to_parquet(output_path_test)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    output_path_test = os.environ.get("output_path_test")
    train_size = float(os.environ.get("train_size", "0.7"))
    stratify = os.environ.get("stratify")
    seed = int(os.environ.get("seed", "0"))

    train_test_split_data(
        input_path=input_path,
        output_path=output_path,
        output_path_test=output_path_test,
        train_size=train_size,
        stratify=stratify,
        seed=seed
    ) 