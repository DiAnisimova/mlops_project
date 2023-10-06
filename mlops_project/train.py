import pandas as pd
from model import RandomForestMSE
import pickle


data = pd.read_csv('train.csv')
y = data['price']
X = data.drop('price', axis=1)
X = X.to_numpy()
y = y.to_numpy()
regr = RandomForestMSE(n_estimators=500)
regr.fit(X, y)
with open('model.pkl','wb') as f:
    pickle.dump(regr, f)
