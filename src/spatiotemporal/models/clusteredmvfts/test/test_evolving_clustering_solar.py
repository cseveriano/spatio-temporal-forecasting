import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from spatiotemporal.models.clusteredmvfts.fts import evolvingclusterfts


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

df = pd.read_pickle(os.path.join(os.getcwd(), "../../../../../notebooks/df_oahu.pkl"))

## Remove columns with many corrupted or missing values
df.drop(columns=['AP_1', 'AP_7'], inplace=True)

#Normalize Data

# Save Min-Max for Denorm
min_raw = df[target_station].min()

max_raw = df[target_station].max()

# Perform Normalization
norm_df = normalize(df)

# Split data
#interval = ((df.index >= '2010-06') & (df.index < '2010-12'))
interval = ((df.index >= '2010-11') & (df.index <= '2010-12'))

(train_df, validation_df, test_df) = split_data(df, interval)
(norm_train_df, norm_validation_df, norm_test_df) = split_data(norm_df, interval)



#####


def evol_cluster_forecast(train_df, test_df):

    model = evolvingclusterfts.EvolvingClusterFTS(t_norm='nonzero', defuzzy='weighted')

    model.fit(train_df.values, order=_order, verbose = False)

    forecast = model.predict(test_df.values)
    forecast_df = pd.DataFrame(data=forecast, columns=test_df.columns)
    return forecast_df

steps = 1
_order = 2

forecast = evol_cluster_forecast(norm_train_df[input], norm_validation_df[input])

forecast = denormalize(forecast[output], df[output])

rmse = calculate_rmse(validation_df[output], forecast, _order, steps)
print("RMSE: ", rmse)

plt.figure()
plt.plot(validation_df[output].iloc[_order:600].values)
plt.plot(forecast[:(600-_order)])
plt.show()
