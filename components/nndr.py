import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np


def nndr(
    input_path: str,
    output_path: str,
    input_path_sd: str,
    discrete_features: list,
    percentiles: list,
):
    """
    Evaluates Nearest Neighbour Distance Ratio metric.
    Returns results at various percentiles, we suggest, e.g., 1, 2, 10, 25, 50.
    """
    rd = pd.read_parquet(input_path)
    sd = pd.read_parquet(input_path_sd)

    # preprocessing for evaluation in adequate feature space (standardizing and one hot encoding)
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(pd.concat((rd[discrete_features], sd[discrete_features])))
    ohe_rd = pd.DataFrame(
        ohe.transform(rd[discrete_features]),
        columns=ohe.get_feature_names_out(),
        index=rd.index,
    )
    ohe_sd = pd.DataFrame(
        ohe.transform(sd[discrete_features]),
        columns=ohe.get_feature_names_out(),
        index=rd.index,
    )
    rd = rd.drop(discrete_features, axis=1).join(ohe_rd)
    sd = sd.drop(discrete_features, axis=1).join(ohe_sd)

    scaler = StandardScaler()
    numerical_features = rd.columns.difference(discrete_features)
    rd[numerical_features] = scaler.fit_transform(rd[numerical_features])
    sd[numerical_features] = scaler.transform(sd[numerical_features])

    # get nearest neighbour distance ratios
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(rd)
    distances, indices = nn.kneighbors(sd)
    nearest_dist = distances[:, 0]
    second_nearest_dist = distances[:, 1]
    nndr = nearest_dist / second_nearest_dist

    # compute percentile scores
    percentiles = sorted(percentiles)
    score = pd.DataFrame(
        [np.percentile(nndr, percentiles)],
        columns=[f"NNDR (percentile {x})" for x in percentiles],
    )
    score.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    input_path_sd = os.environ.get("input_path_sd")
    discrete_features = os.environ.get("discrete_features")
    percentiles = os.environ.get("percentiles")

    nndr(
        input_path=input_path,
        output_path=output_path,
        input_path_sd=input_path_sd,
        discrete_features=discrete_features,
        percentiles=percentiles,
    )
