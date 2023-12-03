import pickle
from io import StringIO

import dvc.api
import pandas as pd
from sklearn.metrics import mean_squared_error


def infer(**kwargs):
    with open("model.pkl", "rb") as f:
        regr = pickle.load(f)
    data = dvc.api.read(path="./data/test.csv", mode="r")
    data = pd.read_csv(StringIO(data))
    y = data[kwargs["target"]]
    X = data.drop(kwargs["target"], axis=1)
    X = X.to_numpy()
    y = y.to_numpy()
    pred = regr.predict(X)
    print("RMSE =", mean_squared_error(y, pred, squared=False))
    res = pd.DataFrame(data=pred, columns=[kwargs["column_name"]])
    res.to_csv(kwargs["prediction_name"], index=False)


if __name__ == "__main__":
    infer()
