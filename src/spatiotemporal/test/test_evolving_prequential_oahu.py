import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from spatiotemporal.models.clusteredmvfts.fts import evolvingclusterfts
from spatiotemporal.util import sampling
import matplotlib.pyplot as plt
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
        print("processing: " + str(test_end) + " of " + str(limit))
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


_percent = 0.1
_window_size = round(len(single_day_df.index) * _percent)
_order = 2
_step = 1
_input = ['DH4', 'DH5', 'DH6']
#_input = ['DH4', 'DH5']
_output = ['DH4']

model = evolvingclusterfts.EvolvingClusterFTS(t_norm='nonzero', defuzzy='weighted', variance_limit=0.001)

model.fit(initial_train_df[_input].values, order=_order)

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
accumulated_error, error_list, fcst = prequential_evaluation(model, single_day_df, _input, _output, _order, _step, _window_size, _window_size)

plt.figure()
plt.plot(single_day_df[_output].iloc[_order:].values)
plt.plot(fcst[:-_order])
plt.show()