import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np


def nndr(
    input_path_train: str,
    output_path: str,
    input_path_test: str,
    input_path_sd: str,
    discrete_features: list,
    percentiles: list,
):
    """
    Evaluates Nearest Neighbour Distance Ratio metric.
    Returns results at various percentiles, we suggest, e.g., 1, 2, 10, 25, 50.
    """
    train = pd.read_parquet(input_path_train)
    test = pd.read_parquet(input_path_test)
    sd = pd.read_parquet(input_path_sd)

    # preprocessing
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(
        pd.concat(
            (train[discrete_features], test[discrete_features], sd[discrete_features])
        )
    )

    def _onehot_encode(df):
        onehot = pd.DataFrame(
            ohe.transform(df[discrete_features]),
            columns=ohe.get_feature_names_out(),
            index=df.index,
        )
        df = df.drop(discrete_features, axis=1).join(onehot)
        return df

    train, test, sd = map(_onehot_encode, [train, test, sd])

    scaler = StandardScaler()
    numerical_features = train.columns.difference(discrete_features)
    scaler.fit(train[numerical_features])

    def _standard_scale(df):
        df[numerical_features] = scaler.transform(df[numerical_features])
        return df

    train, test, sd = map(_standard_scale, [train, test, sd])

    def _get_nndr(df1, df2):
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(df1)
        distances, indices = nn.kneighbors(df2)
        nndr = distances[:, 0] / distances[:, 1]
        return nndr

    nndr_train = _get_nndr(sd, train)
    nndr_test = _get_nndr(sd, test)

    # compute percentile scores
    percentiles = sorted(percentiles)
    score = pd.DataFrame(
        [
            np.percentile(nndr_train, percentiles),
            np.percentile(nndr_test, percentiles),
            np.percentile(nndr_train, percentiles)
            / np.percentile(nndr_test, percentiles),
        ],
        index=["NNDR train", "NNDR holdout", "NNDR ratio"],
        columns=[f"Percentile {x}" for x in percentiles],
    )
    score.to_parquet(output_path)


if __name__ == "__main__":
    input_path_train = os.environ.get("input_path_train")
    output_path = os.environ.get("output_path")
    input_path_test = os.environ.get("input_path_test")
    input_path_sd = os.environ.get("input_path_sd")
    discrete_features = os.environ.get("discrete_features")
    percentiles = os.environ.get("percentiles")

    nndr(
        input_path_train=input_path_train,
        output_path=output_path,
        input_path_test=input_path_test,
        input_path_sd=input_path_sd,
        discrete_features=discrete_features,
        percentiles=percentiles,
    )
