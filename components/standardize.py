import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize(
    input_path: str,
    output_path: str,
    columns: list,
):
    """
    Performs z-score standardization of numerical columns.
    TBD: encoder should be fit on training data only to avoid leakage to test set.
    """
    df = pd.read_parquet(input_path)

    encoder = StandardScaler()
    df[columns] = encoder.fit_transform(df[columns])

    df.to_parquet(output_path)


if __name__ == "__main__":
    columns = os.environ.get("columns", "").split(",")
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")

    standardize(input_path=input_path, output_path=output_path, columns=columns) 