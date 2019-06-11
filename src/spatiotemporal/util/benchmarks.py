import pandas as pd
from pyFTS.benchmarks import Measures

def __pop(key, default, kwargs):
    if key in kwargs:
        return kwargs.pop(key)
    else:
        return default

def resample_data(df, frequency):
  return df.resample(frequency).mean()


def train_test_split(df, training_length):
    # number of days for training
    limit = round(len(pd.unique(df.index.date)) * training_length)
    ds = pd.Series(df.index.date)

    training_days = pd.unique(df.index.date)[:limit]
    train_df = df.loc[ds.isin(training_days).values, :]

    testing_days = pd.unique(df.index.date)[limit:]
    test_df = df.loc[ds.isin(testing_days).values, :]

    return train_df, test_df

def rolling_window_benchmark(data, train=0.8, **kwargs):
    resample = __pop('resample', None, kwargs)

    if resample:
        data = resample_data(data, resample)

    train_data, test_data = train_test_split(data, train)

    methods = __pop('methods', None, kwargs)
    order = __pop("order", [1, 2, 3], kwargs)
    steps_ahead = __pop('steps_ahead', [1], kwargs)

    for method in methods:
        method.fit(train_data, **kwargs)

        #_start = time.time()
        _rmse, _smape, _u = Measures.get_point_statistics(test_data, method, **kwargs)
        #_end = time.time()

