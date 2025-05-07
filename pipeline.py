import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import all_estimators
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.base import is_classifier, is_regressor


def _standardize(input_path: str, output_path: str, columns: list):
    """ "
    Performs z-score standardization of numerical columns.
    TBD: encoder should be fit on training data only to avoid leakage to test set.
    """

    df = pd.read_parquet(input_path)

    encoder = StandardScaler()
    df[columns] = encoder.fit_transform(df[columns])

    df.to_parquet(output_path)


def _ml_experiment(
    input_path: str,
    output_path: str,
    input_path_test: str,
    target_col: str,
    model_name: str = "LogisticRegression",
    model_hparams: dict = {},
):
    """
    Performs an ML experiment using a prediction model from sklearn.
    Provides ROCAUC for classification, RMSE for regression.
    """

    train = pd.read_parquet(input_path)
    test = pd.read_parquet(input_path_test)

    y_tr, y_te = train[target_col], test[target_col]
    X_tr, X_te = train.drop(target_col, axis=1), test.drop(target_col, axis=1)

    models = dict(all_estimators())
    model = models[model_name]
    model = model(**model_hparams)

    model.fit(X_tr, y_tr)
    if is_classifier(model):
        preds = model.predict_proba(X_te)
        if y_tr.nunique() == 2:
            score = roc_auc_score(y_te, preds[:, 1])
        else:
            score = roc_auc_score(y_te, preds, multi_class="ovr", average="micro")
        score = pd.DataFrame([score], columns=["ROCAUC"])
    elif is_regressor(model):
        preds = model.predict(X_te)
        score = root_mean_squared_error(y_te, preds)
        score = pd.DataFrame([score], columns=["RMSE"])
    score.to_parquet(output_path)


def _train_test_split(
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

    df_tr, df_te = train_test_split(
        df, train_size=train_size, stratify=df[stratify], random_state=seed
    )

    df_tr.to_parquet(output_path)
    df_te.to_parquet(output_path_test)


def _drop_nan(
    input_path: str,
    output_path: str,
):
    """
    Drop na values from a dataframe.
    """

    df = pd.read_parquet(input_path)

    df = df.dropna().reset_index(drop=True)

    df.to_parquet(output_path)


def _one_hot(input_path: str, output_path: str, columns: list):
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


def _to_numeric(input_path: str, output_path: str, columns: list):
    """
    Coerces columns to numeric dtype.
    Sets as NaN if not possible.
    """
    df = pd.read_parquet(input_path)
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_parquet(output_path)


def _feature_selection(input_path: str, output_path: str, columns: list):
    """
    Manually select a set of features to retain.
    """
    df = pd.read_parquet(input_path)
    df = df[columns]
    df.to_parquet(output_path)


def _date_to_numeric(input_path: str, output_path: str, columns: list):
    """
    Cast datetime columns to a numeric format, i.e., (fractional) proleptic Gregorian ordinal.
    This makes downstream feature engineering and modelling much easier, as most data processing cannot handle datetimes.
    """
    df = pd.read_parquet(input_path)

    def date_to_floating_ordinal(dt):
        """
        Calculates fractional days of datetime, as pd.toordinal() rounds to full days.
        """
        return (
            dt.toordinal()
            + (
                dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)
            ).total_seconds()
            / 86400
        )

    for col in columns:
        df[col] = pd.to_datetime(df[col])
        df[col] = df[col].apply(date_to_floating_ordinal)

    df.to_parquet(output_path)


def _engineer_feature(
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


def _expand_feature(input_path: str, output_path: str, feature: str, expand_on: str):
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


def _cast_dtypes(input_path: str, output_path: str, dtype_map: dict):
    """
    Cast features to their correct dtype.
    Often an important step for further downstream feature engineering.
    Excepts dtype_map as a dictionary, e.g.: {col1:"float", col2:"str", col3:"datetime"}
    """
    df = pd.read_parquet(input_path)
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                if "datetime" in dtype:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                raise ValueError(f"Could not cast column '{col}' to '{dtype}': {e}")
        else:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    df.to_parquet(output_path)


def _merge_on_date(
    input_path: str,
    output_path: str,
    input_path_merge: str,
    date_col: str,
    date_col_merge: str,
    merge_on: str = None,
    direction: str = "nearest",
):
    """
    Merge on closest date (options are "nearest",  "backward", "forward").
    Useful for adding data to a cohort which was measured closest to a certain event, e.g., admission or death.
    Optionally merge on merge_on column first to, e.g., ensure measurements are matched for the correct patient or visit.
    """

    df = pd.read_parquet(input_path)
    df_merge = pd.read_parquet(input_path_merge)

    df[date_col] = pd.to_datetime(df[date_col])
    df_merge[date_col_merge] = pd.to_datetime(df_merge[date_col_merge])
    df = df.sort_values(date_col)
    df_merge = df_merge.sort_values(date_col_merge)

    df = pd.merge_asof(
        df,
        df_merge,
        by=merge_on,
        left_on=date_col,
        right_on=date_col_merge,
        direction=direction,
    )

    df.to_parquet(output_path)


def _retrieve_col_val(
    input_path: str,
    output_path: str,
    retrieve_string: str,
    match_type: str = "startswith",
    name_col: str = "result_name",
    val_col: str = "result_value",
):
    """
    Change table layout to col:val instead of col1:col_names,col2:col_values.
    Can be used to retrieve specific measurements from tables which store various measurement types in a single column.
    """

    df = pd.read_parquet(input_path)

    if match_type == "startswith":
        df = df[df[name_col].str.startswith(retrieve_string)]
    elif match_type == "endswith":
        df = df[df[name_col].str.endswith(retrieve_string)]
    elif match_type == "full_match":
        df = df[df[name_col] == retrieve_string]
    else:
        raise Exception("Invalid match type.")

    df = df.rename({val_col: retrieve_string}, axis=1)
    df = df.drop(name_col, axis=1)

    df.to_parquet(output_path)


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
