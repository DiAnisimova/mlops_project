import pickle
from io import StringIO

import dvc.api
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error


def infer(**kwargs):
    data = dvc.api.read(path="./data/test.csv", mode="r")
    data = pd.read_csv(StringIO(data))
    y = data[kwargs["target"]]
    X = data.drop(kwargs["target"], axis=1)
    X = X.to_numpy()
    y = y.to_numpy()

    if "uri" in kwargs:
        mlflow.set_tracking_uri(uri=kwargs["uri"])
        regr = mlflow.pyfunc.load_model(kwargs["model_uri"])
    with open("model.pkl", "rb") as f:
        regr = pickle.load(f)

    pred = regr.predict(X)
    print("RMSE =", mean_squared_error(y, pred, squared=False))
    res = pd.DataFrame(data=pred, columns=[kwargs["column_name"]])
    res.to_csv(kwargs["prediction_name"], index=False)


if __name__ == "__main__":
    param_dict = dict()
    param_dict["target"] = "price"
    param_dict["column_name"] = "pred_price"
    param_dict["prediction_name"] = "prediction.csv"
    infer(param_dict)
