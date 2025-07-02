import os
import pandas as pd
import numpy as np
from scipy import stats


def compute_mixed_correlation_matrix(data: pd.DataFrame, discrete_features: list):
    numerical_features = [x for x in data.columns if x not in discrete_features]

    data[discrete_features] = data[discrete_features].astype(str)
    data[numerical_features] = data[numerical_features].astype(float)

    corr_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)

    for i, f1 in enumerate(data.columns):
        for j, f2 in enumerate(data.columns):
            if j < i:
                continue  # Only compute upper triangular part

            if f1 in numerical_features and f2 in numerical_features:
                corr_matrix.loc[f1, f2] = stats.spearmanr(data[f1], data[f2])[0]
            elif f1 in discrete_features and f2 in discrete_features:
                corr_matrix.loc[f1, f2] = cramers_v(data[f1], data[f2])
            elif f1 in numerical_features and f2 in discrete_features:
                corr_matrix.loc[f1, f2] = correlation_ratio(data[f2], data[f1])
            elif f1 in discrete_features and f2 in numerical_features:

                corr_matrix.loc[f1, f2] = correlation_ratio(data[f1], data[f2])

            corr_matrix.loc[f2, f1] = corr_matrix.loc[f1, f2]  # Fill symmetric value

    return corr_matrix.astype(float)


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def correlation_ratio(categories, values):
    categories = np.array(categories)
    values = np.array(values)

    class_means = [values[categories == cat].mean() for cat in np.unique(categories)]
    overall_mean = values.mean()

    numerator = np.sum(
        [
            len(values[categories == cat]) * (class_mean - overall_mean) ** 2
            for cat, class_mean in zip(np.unique(categories), class_means)
        ]
    )
    denominator = np.sum((values - overall_mean) ** 2)

    return numerator / denominator if denominator != 0 else 0


def correlation_matrix(input_path: str, output_path: str, discrete_features: list):
    """
    Return a correlation matrix with:
    - Spearman's r for numerical correlations
    - Cramérs V for discrete correlations
    - Correlation ratio eta for mixed correlations
    """
    df = pd.read_parquet(input_path)
    corr = compute_mixed_correlation_matrix(df, discrete_features=discrete_features)
    corr.to_parquet(output_path)


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    discrete_features = os.environ.get("discrete_features")

    correlation_matrix(
        input_path=input_path,
        output_path=output_path,
        discrete_features=discrete_features,
    )
