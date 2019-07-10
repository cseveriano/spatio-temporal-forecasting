import pickle
from hyperopt import space_eval
from spatiotemporal.util import sampling
import matplotlib.pyplot as plt
from pyFTS.benchmarks import Measures


def forecast_best_params(data, train_split, method_id, method, space, plot=False, save=False):
    print("Running experiment ", method_id)

    best = pickle.load(open("best_" + method_id + ".pkl", "rb"))
    train, test = sampling.train_test_split(data, train_split)
    best_params = space_eval(space, best)
    fcst = method(train, test, best_params)
    _order = best_params['order']
    _output = best_params['output']
    yobs = test[_output].iloc[_order:].values

    if plot:
        plt.figure(figsize=(20, 10))
        plt.plot(yobs)
        plt.plot(fcst)
        plt.show()

    rmse = Measures.rmse(yobs, fcst)
    print("RMSE: ", rmse)

    smape = Measures.smape(yobs, fcst)
    print("SMAPE: ", smape)

    u = Measures.UStatistic(yobs, fcst)
    print("U Statistic: ", u)

    if save:
        results = {"method_id": method_id, "forecast": fcst, "RMSE": rmse, "SMAPE": smape, "U": u}
        pickle.dump(results, open("results_" + method_id + ".pkl", "wb"))

    return rmse, smape, u


def load_best_params(method_id, space, print=False):
    best = pickle.load(open("best_" + method_id + ".pkl", "rb"))
    best_params = space_eval(space, best)
    if print:
        print(best_params)
    return best_params


def forecast_params(data, train_split, method, params, plot=False):
    train, test = sampling.train_test_split(data, train_split)
    fcst = method(train, test, params)
    _output = params['output']
    _offset = params['order'] + params['step'] - 1
    yobs = test[_output].iloc[_offset:].values

    if plot:
        plt.figure(figsize=(20, 10))
        plt.plot(yobs)
        plt.plot(fcst)
        plt.show()

    rmse = Measures.rmse(yobs, fcst)
    print("RMSE: ", rmse)

    smape = Measures.smape(yobs, fcst)
    print("SMAPE: ", smape)

    u = Measures.UStatistic(yobs, fcst)
    print("U Statistic: ", u)

    return rmse, smape, u

import pandas as pd

def rolling_window_forecast_params(data, train_percent, window_size, method, params):

    # get training days
    training_days = pd.unique(data.index.date)
    fcst = []
    yobs = []

    for day in training_days:
        daily_data = data[data.index.date == day]
        nsamples = len(daily_data.index)
        train_size = round(nsamples * train_percent)
        test_end = 0
        index = 0

        while test_end < nsamples:
            train_start, train_end, test_start, test_end = get_data_index(index, train_size, window_size, nsamples)
            train = data[train_start:train_end]
            test = data[test_start:test_end]
            index += window_size

            fcst.extend(method(train, test, params))
            _output = params['output']
            _offset = params['order'] + params['step'] - 1
            yobs.extend(test[_output].iloc[_offset:].values)

    rmse = Measures.rmse(yobs, fcst)
    print("RMSE: ", rmse)

    smape = Measures.smape(yobs, fcst)
    print("SMAPE: ", smape)

    u = Measures.UStatistic(yobs, fcst)
    print("U Statistic: ", u)

    return rmse, smape, u

def get_data_index(index, train_size, window_size, limit):

    train_start = index
    train_end = index + train_size

    test_start = train_end
    test_end = min(test_start + window_size, limit)

    return train_start, train_end, test_start, test_end
