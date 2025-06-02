import os
import pandas as pd


def engineer_feature(
    input_path: str,
    output_path: str,
    feature1: str,
    feature2: str,
    operation: str = "add",
):
    """
    Create a new feature by adding, subtracting, or multiplying two existing features.
    The new feature name will be a contraction of the input feature names.
    """
    df = pd.read_parquet(input_path)

    new_feature = f"{feature1}_{operation}_{feature2}"

    if operation == "add":
        df[new_feature] = df[feature1] + df[feature2]
    elif operation == "subtract":
        df[new_feature] = df[feature1] - df[feature2]
    elif operation == "multiply":
        df[new_feature] = df[feature1] * df[feature2]

    df.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    feature1 = os.environ.get("feature1")
    feature2 = os.environ.get("feature2")
    operation = os.environ.get("operation", "add")

    engineer_feature(
        input_path=input_path,
        output_path=output_path,
        feature1=feature1,
        feature2=feature2,
        operation=operation
    ) 