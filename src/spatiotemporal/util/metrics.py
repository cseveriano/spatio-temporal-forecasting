from pyFTS.benchmarks import Measures
import numpy as np
import math
from sklearn.metrics import mean_squared_error

def normalized_rmse(targets, forecasts):
    if isinstance(targets, list):
        targets = np.array(targets)

    return Measures.rmse(targets, forecasts) / np.nanmean(targets)

def calculate_rmse(test, forecast, order, step):
    rmse = math.sqrt(mean_squared_error(test[(order):], forecast[:-step]))
    return rmse
