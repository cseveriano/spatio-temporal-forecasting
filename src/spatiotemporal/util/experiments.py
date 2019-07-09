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
