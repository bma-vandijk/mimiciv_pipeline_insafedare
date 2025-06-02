import os
import json
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.base import is_classifier, is_regressor


def ml_experiment(
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


if __name__ == "__main__":
    input_path = os.environ.get("input_path")
    output_path = os.environ.get("output_path")
    input_path_test = os.environ.get("input_path_test")
    target_col = os.environ.get("target_col")
    model_name = os.environ.get("model_name", "LogisticRegression")
    model_hparams = json.loads(os.environ.get("model_hparams", "{}"))

    ml_experiment(
        input_path=input_path,
        output_path=output_path,
        input_path_test=input_path_test,
        target_col=target_col,
        model_name=model_name,
        model_hparams=model_hparams
    ) 