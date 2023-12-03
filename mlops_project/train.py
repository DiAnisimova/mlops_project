import os
import pickle
from copy import copy
from io import StringIO

import dvc.api
import git
import mlflow
import onnx
import pandas as pd
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from mlflow.onnx import log_model
from sklearn.model_selection import train_test_split


def train(**kwargs_train):
    commit_id = git.Repo("./").head.object.hexsha
    params = copy(kwargs_train)
    params["commit_id"] = commit_id
    model_name = kwargs_train["model_name"]
    target_name = kwargs_train["target"]
    uri = kwargs_train["uri"]
    del kwargs_train["model_name"]
    del kwargs_train["target"]
    del kwargs_train["uri"]
    data = dvc.api.read(path="./data/train.csv", mode="r")
    data = pd.read_csv(StringIO(data))
    y = data[target_name]
    X = data.drop(target_name, axis=1)
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.8, random_state=42
    )
    kwargs_train["custom_metric"] = ["RMSE", "MAE", "R2"]

    regr = CatBoostRegressor(**kwargs_train)
    regr.fit(X=X_train, y=y_train, eval_set=(X_val, y_val))
    if not os.path.exists("./models/"):
        os.mkdir("./models/")
    with open("./models/" + model_name + ".pkl", "wb") as f:
        pickle.dump(regr, f)
    with open("model.pkl", "wb") as f:
        pickle.dump(regr, f)

    regr.save_model(
        "housing.onnx",
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": "test model for Regression",
            "onnx_graph_name": "CatBoostModel_for_Regression",
        },
    )
    onnx_model = onnx.load_model("housing.onnx")

    train_info = pd.read_csv("./catboost_info/learn_error.tsv", sep="\t")
    val_info = pd.read_csv("./catboost_info/test_error.tsv", sep="\t")
    mlflow.set_tracking_uri(uri=uri)

    # # Create a new MLflow Experiment
    mlflow.set_experiment(model_name)

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        for i, val in enumerate(train_info["RMSE"]):
            mlflow.log_metric("rmse train", val, step=i)
        for i, val in enumerate(val_info["RMSE"]):
            mlflow.log_metric("rmse val", val, step=i)
        for i, val in enumerate(train_info["R2"]):
            mlflow.log_metric("r2_score train", val, step=i)
        for i, val in enumerate(val_info["R2"]):
            mlflow.log_metric("r2_score val", val, step=i)
        mlflow.set_tag("Training Info", "Catboost model for housing data")

        # Infer the model signature
        signature = infer_signature(X_train, regr.predict(X_train))
        model_info = log_model(onnx_model, model_name, signature=signature)
        with open("./configs/server/cfg.yaml", "w") as f:
            f.write('target: "price"\n')
            f.write('column_name: "pred_price"\n')
            f.write('prediction_name: "prediction.csv"\n')
            f.write("model_uri: " + '"' + model_info.model_uri + '"' + "\n")
            f.write('uri: "http://127.0.0.1:9000"')
        # onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
        # print("UUUUUUU", model_info.model_uri)

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=regr,
            artifact_path=model_name + "_info",
            signature=signature,
            input_example=X_train,
            registered_model_name=model_name,
        )


if __name__ == "__main__":
    train()
