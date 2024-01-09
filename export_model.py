from copy import copy
from io import StringIO

import dvc.api
import git
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split


def train(**kwargs_train):
    commit_id = git.Repo("./").head.object.hexsha
    params = copy(kwargs_train)
    params["commit_id"] = commit_id
    target_name = kwargs_train["target"]
    del kwargs_train["model_name"]
    del kwargs_train["target"]
    del kwargs_train["uri"]
    data = dvc.api.read(path="./data/train.csv", mode="r")
    data = pd.read_csv(StringIO(data))
    y = data[target_name]
    X = data.drop(target_name, axis=1)
    X = X.to_numpy()
    y = y.to_numpy()[:, None]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.8, random_state=42
    )
    kwargs_train["custom_metric"] = ["RMSE", "MAE", "R2"]

    regr = CatBoostRegressor(**kwargs_train)
    batch_train = Pool(X_train, label=y_train)
    batch_val = Pool(X_val, label=y_val)
    regr.fit(X=batch_train, eval_set=batch_val)
    # regr.fit(X=X_train, y=y_train, eval_set=(X_val, y_val))
    # if not os.path.exists("./models/"):
    #     os.mkdir("./models/")
    # with open("./models/" + model_name + ".pkl", "wb") as f:
    #     pickle.dump(regr, f)
    # with open("model.pkl", "wb") as f:
    #     pickle.dump(regr, f)

    regr.save_model(
        "./model_repository/catboost_onnx/1/model.onnx",
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": "test model for Regression",
            "onnx_graph_name": "CatBoostModel_for_Regression",
        },
    )
    # onnx.load_model("./model_repository/catboost_onnx/1/model.onnx")


if __name__ == "__main__":
    train(
        **{
            "iterations": 1000,
            "model_name": "model0",
            "target": "price",
            "uri": "http://127.0.0.1:9000",
        }
    )
