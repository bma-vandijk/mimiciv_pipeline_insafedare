import os
import pandas as pd


def expand_feature(
    input_path: str,
    output_path: str,
    feature: str,
    expand_on: str,
):
    """
    Expands columns which contain multiple entries.
    For example, Blood Pressure might contain both systolic and diastolic blood pressure in a single column.
    """
    df = pd.read_parquet(input_path)
    df[feature] = df[feature].astype(str)
    expanded_features = df[feature].str.split(expand_on, expand=True)
    expanded_feature_names = [
        f"{feature}_" + str(x) for x in range(expanded_features.shape[1])
    ]
    df[expanded_feature_names] = expanded_features
    df = df.drop(feature, axis=1)
    
    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    feature = os.environ.get("feature")
    expand_on = os.environ.get("expand_on")

    expand_feature(
        input_path=input_path,
        output_path=output_path,
        feature=feature,
        expand_on=expand_on
    ) 