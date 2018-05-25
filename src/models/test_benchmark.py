from sklearn.cluster import MeanShift, estimate_bandwidth

import pandas as pd
import numpy as np
from models import KMeansPartitioner
from sklearn import preprocessing
import pyFTS.benchmarks as bchmk
from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.models import hofts
from pyFTS.common import Util as cUtil

from models import sthofts

df = pd.read_pickle("../../notebooks/df_oahu.pkl")
df_ssa_clean = pd.read_pickle("../../notebooks/df_ssa_clean.pkl")
df_ssa_residual = pd.read_pickle("../../notebooks/df_ssa_residual.pkl")


sample_df = df_ssa_residual.loc['2010-11']

week = (sample_df.index.day - 1) // 7 + 1
# PARA OS TESTES:
# 2 SEMANAS PARA TREINAMENTO
train_df = sample_df.loc[week <= 2]

# 1 SEMANA PARA VALIDACAO
validation_df = sample_df.loc[week == 3]

# 1 SEMANA PARA TESTES
test_df = sample_df.loc[week > 3]


# FAZER A BUSCA PELO MELHOR PARAMETRO
# COMPARAR O VALOR COM HOFTS
# DEFNIDO O MELHOR PARAMETRO, TESTAR O CASO ANUAL


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

model_hofts = sthofts.SpatioTemporalHighOrderFTS("FTS", nlags=_order, partitioner=fuzzy_sets)
model_hofts.fit(train, dump = 'time', num_batches=100, distributed=False, nodes=['192.168.1.3'])
cUtil.persist_obj(model_hofts, "sthofts.pkl")

model_hofts = cUtil.load_obj("sthofts.pkl")

forecast_hofts = model_hofts.predict(validation)