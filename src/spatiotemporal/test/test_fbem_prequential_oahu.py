import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from spatiotemporal.models.clusteredmvfts.fts import evolvingclusterfts
from spatiotemporal.util import sampling
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import concat
from fbem.FBeM import FBeM
import fbem.graph_utils as fbgu

import warnings

#warnings.filterwarnings('error')

def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def calculate_rmse(test, forecast, order, step):
    rmse = math.sqrt(mean_squared_error(test[(order):], forecast[:-step]))
    print("RMSE : " + str(rmse))
    return rmse 


def prequential_evaluation(method, df, input_columns, output_columns, order, step, train_size, window_size=1, fading_factor=1):
    test_end = 0
    index = 0
    limit = len(df[output_columns])
    accumulated_error = 0
    error_list = []
    fcst = []
    data_input = df[input_columns].values
    data_output = df[output_columns].values

    while test_end < limit:
        train_start, train_end, test_start, test_end = getDataIndex(index, train_size, window_size, limit)

        if (test_end - test_start) > order:
            print(train_start, train_end, test_start, test_end)
            train_data = data_input[train_start:train_end]
            test_data = data_input[test_start:test_end]
            index = test_start

            method.fit(train_data, order=order)
            y_hat_df = pd.DataFrame(method.predict(test_data), columns=input_columns)
            y_hat = y_hat_df[output_columns].to_numpy()[:, 0]

            fcst.extend(y_hat)
            y = data_output[test_start:test_end]

            error = calculate_rmse(y, y_hat, order, step)
            accumulated_error += fading_factor * error
            error_list.append(error)

    return accumulated_error, error_list, fcst


def getDataIndex(index, train_size, window_size, limit):
    train_start = index

    if index == 0:
        train_end = index + train_size
    else:
        train_end = index + window_size

    test_start = train_end
    test_end = min(test_start + window_size, limit)

    return train_start, train_end, test_start, test_end


hink_raw_df = pd.read_csv("../../../data/processed/NREL/Oahu/hink_raw_df.csv",  parse_dates=['Time'], index_col=0)
hink_cs_df = pd.read_csv("../../../data/processed/NREL/Oahu/hink_cs_df.csv",  parse_dates=['Time'], index_col=0)

# Resample data
hink_raw_df = sampling.resample_data(hink_raw_df, "5min")
hink_cs_df = sampling.resample_data(hink_cs_df, "5min")

# Filter zenith angle
zenith_angle_index = hink_raw_df.zen < 80
hink_raw_df = hink_raw_df[zenith_angle_index]
hink_cs_df = hink_cs_df[zenith_angle_index]

norm_hink_raw_df = normalize(hink_raw_df)
norm_hink_cs_df = normalize(hink_cs_df)

'''
Hinkelman days:
2010-07-31
2010-08-01
2010-08-02
2010-08-03
2010-08-04
2010-08-05
2010-08-21
2010-08-29
2010-09-05
2010-09-06
2010-09-07
2010-09-21
2010-10-27
'''

initial_train_df = norm_hink_raw_df["2010-07-31":"2010-08-03"]
single_day_df = norm_hink_raw_df["2010-08-04"]



#df = pd.read_pickle("../../../data/processed/Pickle/df_oahu.pkl")

## Setting parameters
_percent = 0.1
_window_size = round(len(single_day_df.index) * _percent)
_order = 2
_step = 1
_input = ['DH4', 'DH5', 'DH6']
#_input = ['DH4', 'DH5']
_output = ['DH4']
## Setting parameters


## Preparing dataset
'''
x é formado como:
x = [x11, x21, x12, x22, ..., xij]
onde i é a variável e j é o lag

y = [y1, y2, ..., yn]
'''


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


fbem_data = series_to_supervised(single_day_df[_input], _order, _step)

fbi = FBeM()
fbi.debug = True
to_normalize = 1
fbi.rho = 0.3

for index, row in fbem_data.iterrows():
    y_limit = -(_step*len(_input))
    x = row[:y_limit].tolist()
    y = row[y_limit:y_limit + (len(_output) * _step)].tolist()

    fbi.learn(x=x, y=y)

print("Final RMSE: ", fbi.rmse[len(fbi.rmse) - 1])
print("Final NDEI: ", fbi.ndei[len(fbi.ndei) - 1])

y_obs = single_day_df[_output].iloc[_order:].values
fbgu.plot_singular_output(fbi, y_obs) # Plot singular output
fbgu.plot_show()
'''
fbgu.plot_granular_output(fbi, yavg) # Plot granular output
fbgu.plot_rmse_ndei(fbi)             # Plot RMSE and NDEI
fbgu.plot_rules_number(fbi)          # Plot rules number
fbgu.plot_rho_values(fbi)            # Plot  Rho variation
fbgu.plot_granules_3d_space(fbem_instance=fbi, max=max(yavg), min=min(yavg)) # Plot a granule example
'''

#plot microclusters after fit

'''
micro_clusters = model.partitioner.clusterer.get_all_active_micro_clusters()
ax = plt.gca()

for m in micro_clusters:
    mean = m["mean"]
    std = math.sqrt(m["variance"])

    circle = plt.Circle(mean, std, color='r', fill=False)

    ax.add_artist(circle)
plt.draw()
'''


model = evolvingclusterfts.EvolvingClusterFTS(t_norm='nonzero', defuzzy='weighted', variance_limit=0.001)

_window_size = 1

#model.fit(initial_train_df[_input].values, order=_order)

accumulated_error, error_list, fcst = prequential_evaluation(model, single_day_df, _input, _output, _order, _step, _window_size, _window_size)

plt.figure()
plt.plot(single_day_df[_output].iloc[_order:].values)
plt.plot(fcst[:-_order])
plt.show()