import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot(
    input_path: str,
    output_path: str,
    columns: list,
):
    """
    One hot encodes features, attaches them to the dataframe, and removes original features.
    Useful to numerically encode categorical features.
    """
    df = pd.read_parquet(input_path)

    encoder = OneHotEncoder(sparse_output=False)
    ohe_features = encoder.fit_transform(df[columns])
    ohe_features = pd.DataFrame(
        ohe_features, columns=encoder.get_feature_names_out(columns), index=df.index
    )
    df = df.drop(columns, axis=1)
    df = pd.concat([df, ohe_features], axis=1)

    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    columns = os.environ.get("columns", "").split(",")

    one_hot(
        input_path=input_path,
        output_path=output_path,
        columns=columns
    ) 