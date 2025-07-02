import os
import re
import pandas as pd


def domain_constraint(input_path: str, output_path: str, constraint: str):
    """
    Evaluate fraction of rows which follow an input constraint.
    """
    df = pd.read_parquet(input_path)

    # map whitespaces to underscores in column names to ensure proper evaluation
    col_map = {col: col.replace(" ", "_") for col in df.columns}
    df.columns = col_map.values()
    for orig, new in col_map.items():
        constraint = re.sub(rf"\b{re.escape(orig)}\b", new, constraint)

    # identify column names used in the constraint and ensure numeric dtypes
    tokens = re.findall(r"\b\w+\b", constraint)
    constraint_cols = [x for x in tokens if x in df.columns]
    df[constraint_cols] = df[constraint_cols].astype(float)

    score = df.eval(constraint).mean()
    score = pd.DataFrame([[score]], columns=["Domain Constraint Score"])
    score.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    constraint = os.environ.get("constraint")

    domain_constraint(
        input_path=input_path, output_path=output_path, constraint=constraint
    )
