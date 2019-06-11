import pandas as pd
import numpy as np
import os
from pyFTS.partitioners import Grid, Entropy, Util as pUtil
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import space_eval
import traceback
import pickle

from clusteredmvfts.fts import cmvhofts
from clusteredmvfts.partitioner import EvolvingClusteringPartitioner



def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def denormalize(norm_data, original_data):
    min = original_data.min()

    max = original_data.max()

    return [(n * (max-min)) + min for n in norm_data]

def calculate_rmse(test, forecast, order, step):
    rmse = math.sqrt(mean_squared_error(test.iloc[(order):], forecast[:-step]))
    print("RMSE : "+str(rmse))
    return rmse

def split_data(df, interval):
    sample_df = df.loc[interval]

    week = (sample_df.index.day - 1) // 7 + 1

    # PARA OS TESTES:
    # 2 SEMANAS PARA TREINAMENTO
    train_df = sample_df.loc[week <= 2]

    # 1 SEMANA PARA VALIDACAO
    validation_df = sample_df.loc[week == 3]

    # 1 SEMANA PARA TESTES
    test_df = sample_df.loc[week > 3]

    return (train_df, validation_df, test_df)

#####

#Set target and input variables
target_station = 'DHHL_3'

two_stations = ['DHHL_3',  'DHHL_4']

#All neighbor stations with residual correlation greater than .90
neighbor_stations_90 = ['DHHL_3',  'DHHL_4','DHHL_5','DHHL_10','DHHL_11','DHHL_9','DHHL_2', 'DHHL_6','DHHL_7','DHHL_8']

input = neighbor_stations_90
output = target_station

df = pd.read_pickle(os.path.join(os.getcwd(), "../notebooks/df_oahu.pkl"))

## Remove columns with many corrupted or missing values
df.drop(columns=['AP_1', 'AP_7'], inplace=True)

#Normalize Data

# Perform Normalization
norm_df = normalize(df)

# Split data
interval = ((df.index >= '2010-06') & (df.index < '2010-07'))

(train_df, validation_df, test_df) = split_data(df, interval)
(norm_train_df, norm_validation_df, norm_test_df) = split_data(norm_df, interval)



###### CNN FUNCTIONS ###########
def evol_cluster_forecast(train_df, test_df, params):
    fuzzy_sets = EvolvingClusteringPartitioner.EvolvingClusteringPartitioner(data=train_df, variance_limit=params['variance_limit'], debug=True)

    model = cmvhofts.ClusteredMultivariateHighOrderFTS(t_norm='nonzero', defuzzy='weighted')

    model.fit(train_df.values, order=params['order'], partitioner=fuzzy_sets, verbose = False)

    forecast = model.predict(test_df.values)
    forecast_df = pd.DataFrame(data=forecast, columns=test_df.columns)
    return forecast_df


def evolving_objective(params):
    print(params)
    try:
        forecast = evol_cluster_forecast(norm_train_df, norm_validation_df, params)
        forecast = denormalize(forecast[output], df[output])
        rmse = calculate_rmse(validation_df[output], forecast, params['order'], 1)

    except Exception:
        traceback.print_exc()
        rmse = 1000

    return {'loss': rmse, 'status': STATUS_OK}


###### OPTIMIZATION ROUTINES ###########
space = {'order': hp.choice('order', [2,4,8]),
        'defuzzy': hp.choice('defuzzy', ['weighted', 'mean']),
        'variance_limit': hp.choice('variance_limit', [0.001, 0.002, 0.005, 0.01, 0.05])}


# trials = pickle.load(open("tuning_results.pkl", "rb"))
# best = pickle.load(open("best_result.pkl", "rb"))

trials = Trials()
best = fmin(evolving_objective, space, algo=tpe.suggest, max_evals =500, trials=trials)
print('best: ')
print(space_eval(space, best))

pickle.dump(best, open("best_result.pkl", "wb"))
pickle.dump(trials, open("tuning_results.pkl", "wb"))

