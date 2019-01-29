import pandas as pd
import os
from clusteredmvfts.fts import mvhofts
from pyFTS.common import Util as cUtil
from pyFTS.benchmarks import Measures

os.chdir("/Users/cseveriano/spatio-temporal-forecasting/")

fln_df = pd.read_csv('data/processed/SONDA/FLN-15min.csv', sep=";")
fln_train = fln_df[(fln_df.date >= '2013-11-01') & (fln_df.date <= '2014-10-31')]
fln_test = fln_df[(fln_df.date >= '2014-11-01') & (fln_df.date <= '2015-10-31')]

joi_df = pd.read_csv('data/processed/SONDA/JOI-15min.csv', sep=";")
joi_train = joi_df[(joi_df.date >= '2013-11-01') & (joi_df.date <= '2014-10-31')]
joi_test = joi_df[(joi_df.date >= '2014-11-01') & (joi_df.date <= '2015-10-31')]

sbr_df = pd.read_csv('data/processed/SONDA/SBR-15min.csv', sep=";")
sbr_train = sbr_df[(sbr_df.date >= '2013-11-01') & (sbr_df.date <= '2014-10-31')]
sbr_test = sbr_df[(sbr_df.date >= '2014-11-01') & (sbr_df.date <= '2015-10-31')]

from pyFTS.partitioners import Grid

order = 3
nparts = 20

fuzzysets = []
fuzzysets.append(Grid.GridPartitioner(fln_train.glo_avg,nparts))
fuzzysets.append(Grid.GridPartitioner(joi_train.glo_avg,nparts))
fuzzysets.append(Grid.GridPartitioner(sbr_train.glo_avg,nparts))

d = {'fln_glo_avg':fln_train.glo_avg,'sbr_glo_avg':sbr_train.glo_avg,'joi_glo_avg':joi_train.glo_avg}
data_train = pd.DataFrame(d)
data_train = data_train.dropna(axis=0, how='any')

model_file = "models/fts/multivariate/mvhofts-"+str(order)+"-"+str(nparts)+".pkl"



mvhofts = mvhofts.MultivariateHighOrderFTS("")
mvhofts.train(data_train,fuzzysets,order)
cUtil.persist_obj(mvhofts, model_file)


obj = cUtil.load_obj(model_file)
dt = {'fln_glo_avg':fln_test.glo_avg,'sbr_glo_avg':sbr_test.glo_avg,'joi_glo_avg':joi_test.glo_avg}
data_test = pd.DataFrame(dt)
data_test = data_test.dropna(axis=0, how='any')

ret = obj.forecast(data_test)

print("RMSE: " + str(Measures.rmse(list(data_test.fln_glo_avg[order:]), ret[:-1])))
#print(mvhofts)
