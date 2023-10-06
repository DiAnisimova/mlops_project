from time import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
        self,
        n_estimators,
        max_depth=None,
        feature_subsample_size=None,
        **trees_parameters,
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree.
            If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        self.models = []
        self.ind = []
        self.rmse = []
        self.rmse_test = []
        self.time = []
        start = time()
        predictions = np.zeros((y.shape))
        if y_val is not None:
            predictions_test = np.zeros((y_val.shape))
        for i in range(self.n_estimators):
            ind_features = np.random.choice(
                X.shape[1], size=self.feature_subsample_size, replace=False
            )
            sh0 = X.shape[0]
            ind_obj = np.random.choice(sh0, size=sh0, replace=True)
            model = DecisionTreeRegressor(
                max_depth=self.max_depth, **self.trees_parameters
            )
            model.fit(X[ind_obj, :][:, ind_features], y[ind_obj])
            self.models.append(model)
            self.ind.append(ind_features)
            pred_train = model.predict(X[:, ind_features])
            predictions += pred_train
            self.rmse.append(
                (np.average((y - predictions / (i + 1)) ** 2, axis=0)) ** 0.5
            )
            if X_val is not None:
                pred_test = model.predict(X_val[:, ind_features])
                predictions_test += pred_test
                for_av = (y_val - predictions_test / (i + 1)) ** 2
                self.rmse_test.append((np.average(for_av, axis=0)) ** 0.5)
            self.time.append(time() - start)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict(X[:, self.ind[i]]))
        return np.mean(res, axis=0)
