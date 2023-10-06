import pickle
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd



with open('model.pkl', 'rb') as f:
    regr = pickle.load(f)
data = pd.read_csv('test.csv')
y = data['price']
X = data.drop('price', axis=1)
X = X.to_numpy()
y = y.to_numpy()
pred = regr.predict(X)
print(mean_squared_error(y, pred))
res = pd.DataFrame(data=pred, columns=['pred_price'])
res.to_csv('prediction.csv', index=False)
