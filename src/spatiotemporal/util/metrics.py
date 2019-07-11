from pyFTS.benchmarks import Measures
import numpy as np

def normalized_rmse(targets, forecasts):
    if isinstance(targets, list):
        targets = np.array(targets)

    return Measures.rmse(targets, forecasts) / np.nanmean(targets)

