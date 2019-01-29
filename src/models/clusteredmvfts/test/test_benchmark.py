from sklearn.cluster import MeanShift, estimate_bandwidth

import pandas as pd
import numpy as np
from clusteredmvfts.partitioner import KMeansPartitioner
from pyFTS.benchmarks import Measures

from clusteredmvfts.fts import cmvhofts

#Set target and input variables
target_station = 'DHHL_3'

#All neighbor stations with residual correlation greater than .90
neighbor_stations_90 = ['DHHL_3',  'DHHL_4','DHHL_5','DHHL_10','DHHL_11','DHHL_9','DHHL_2', 'DHHL_6','DHHL_7','DHHL_8']

df = pd.read_pickle("../../notebooks/df_oahu.pkl")
df_ssa_clean = pd.read_pickle("../../notebooks/df_ssa_clean.pkl")
df_ssa_residual = pd.read_pickle("../../notebooks/df_ssa_residual.pkl")

## Remove columns with many corrupted or missing values
df.drop(columns=['AP_1', 'AP_7'], inplace=True)
df_ssa_clean.drop(columns=['AP_1', 'AP_7'], inplace=True)
df_ssa_residual.drop(columns=['AP_1', 'AP_7'], inplace=True)


# Get data form the interval of interest
#interval = ((df.index >= '2010-06') & (df.index < '2010-08'))

interval = '2010-11'

df = df.loc[interval]
df_ssa_clean = df_ssa_clean.loc[interval]
df_ssa_residual = df_ssa_residual.loc[interval]


#Normalize Data

# Save Min-Max for Denorm
min_raw = df[target_station].min()
min_clean = df_ssa_clean[target_station].min()
min_residual = df_ssa_residual[target_station].min()

max_raw = df[target_station].max()
max_clean = df_ssa_clean[target_station].max()
max_residual = df_ssa_residual[target_station].max()

sample_df = df_ssa_residual

week = (sample_df.index.day - 1) // 7 + 1
# PARA OS TESTES:
# 2 SEMANAS PARA TREINAMENTO
train_df = sample_df.loc[week <= 2]

# 1 SEMANA PARA VALIDACAO
validation_df = sample_df.loc[week == 3]

# 1 SEMANA PARA TESTES
test_df = sample_df.loc[week > 3]


sample_df = df_ssa_residual.loc['2010-11']
norm_sample_df = (sample_df-sample_df.min())/(sample_df.max()-sample_df.min())


week = (sample_df.index.day - 1) // 7 + 1
# PARA OS TESTES:
# 2 SEMANAS PARA TREINAMENTO
train_df = norm_sample_df.loc[week <= 2]

# 1 SEMANA PARA VALIDACAO
validation_df = norm_sample_df.loc[week == 3]

# 1 SEMANA PARA TESTES
test_df = norm_sample_df.loc[week > 3]


train = np.array(train_df.values)
validation = np.array(validation_df.values)

k = 20

fuzzy_sets = KMeansPartitioner.KMeansPartitioner(data=train, npart=k, batch_size=1000, init_size=k*3)


_order = 3

model_sthofts = cmvhofts.SpatioTemporalHighOrderFTS()

model_sthofts.fit(train_df.values, num_batches=100, order=_order, partitioner=fuzzy_sets)
forecast = model_sthofts.predict(test_df.values)
forecast_df = pd.DataFrame(data=forecast, columns=test_df.columns)
