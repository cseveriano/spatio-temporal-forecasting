from sklearn.cluster import MeanShift, estimate_bandwidth

import pandas as pd
import numpy as np
from clusteredmvfts.partitioner import KMeansPartitioner
from sklearn import preprocessing
from clusteredmvfts.fts import cmvhofts

def normalized_rmse(targets, forecasts):
    dist = [(a - b) ** 2 for a, b in zip(targets, forecasts)]
    return ((np.sqrt(np.nanmean(dist))) / np.nanmean(targets)) * 100


ms_df = pd.read_pickle("../../notebooks/cluster_all_stations_df.pkl")
test_df = pd.read_pickle("../../notebooks/test_cluster_all_stations_df.pkl")

# data = ms_df.AP_1.values
# fuzzy_sets = Grid.GridPartitioner(data=data, npart=100)
# model_hofts = hofts.HighOrderFTS("FTS", partitioner=fuzzy_sets)
# model_hofts.fit(data, order=6, dump = 'time', num_batches=1000, distributed=True, nodes=['192.168.1.3','192.168.1.8'])

train = ms_df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
train = min_max_scaler.fit_transform(train)

test = test_df.values
test = min_max_scaler.fit_transform(test)

k = 4

fuzzy_sets = KMeansPartitioner.KMeansPartitioner(data=train, npart=k, batch_size=1000, init_size=k*3)


_order = 2

import cProfile


model_hofts = cmvhofts.ClusteredMultivariateHighOrderFTS("FTS", nlags=_order, partitioner=fuzzy_sets)

cProfile.run('model_hofts.fit(train, dump = \'time\', num_batches=100)', 'modelfit.profile')

import pstats
stats = pstats.Stats('modelfit.profile')
stats.strip_dirs().sort_stats('time').print_stats()

#model_hofts.fit(train, dump = 'time', num_batches=100)
#model_hofts.fit(train, dump = 'time', num_batches=100, distributed=True, nodes=['192.168.1.3','192.168.1.8'])
#model_hofts.fit(train, dump = 'time', num_batches=100, distributed=True, nodes=['192.168.1.3'])


forecast_hofts = model_hofts.predict(test)


_nrmse = normalized_rmse(test.tolist()[(_order - 1):], forecast_hofts)
print("nRMSE: ", _nrmse, "\n")


# bchmk.sliding_window_benchmarks(dataset, 1000, train=0.8, inc=0.2,
# methods=[pwfts.ProbabilisticWeightedFTS],
# benchmark_models=False,
# transformations=[None],
# orders=[1, 2, 3],
# partitions=np.arange(5, 100, 5),
# progress=False, type='point',
# distributed=True, nodes=['192.168.0.110', '192.168.0.107','192.168.0.106'],
# file="benchmarks.db", dataset="TAIEX", tag="comparisons")
#
#
# #Comando a ser executado em cada maquina para o processo distribuido
# sudo python3 /usr/local/lib/python3.6/dist-packages/dispy/dispynode.py -i 192.168.0.110 -c 3 -d
#
# # Rodar isso na maquina mestre
# ssh -nNT -R 51347:localhost:51347 192.168.0.106
# # É necessario instalar um servidor ssh em cada maquina remota