from sklearn.cluster import MeanShift, estimate_bandwidth

import pandas as pd
import numpy as np
from models import KMeansPartitioner
from sklearn import preprocessing

from pyFTS.models import hofts

def normalized_rmse(targets, forecasts):
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(forecasts, list):
        forecasts = np.array(forecasts)
    return ((np.sqrt(np.nanmean((targets - forecasts) ** 2))) / np.nanmean(targets) ) * 100


ms_df = pd.read_pickle("../../notebooks/cluster_all_stations_df.pkl")
test_df = pd.read_pickle("../../notebooks/test_cluster_all_stations_df.pkl")

train = ms_df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
train = min_max_scaler.fit_transform(train)

test = test_df.values
test = min_max_scaler.fit_transform(test)

k = 20

fuzzy_sets = KMeansPartitioner.KMeansPartitioner(data=train, npart=k, batch_size=1000, init_size=k*3)


_order = 6

model_hofts = hofts.HighOrderFTS("FTS", partitioner=fuzzy_sets)
model_hofts.fit(train, order=_order, dump = 'time')

forecast_hofts = model_hofts.predict(test)
_nrmse = normalized_rmse(test.tolist()[(_order - 1):], forecast_hofts)
print("nRMSE: ", _nrmse, "\n")