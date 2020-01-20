import pandas as pd
from pyFTS.benchmarks import Measures
from pyFTS.common import fts
from pyFTS.partitioners import Grid, Entropy, Util as pUtil
from spatiotemporal.util import sampling, metrics

def __pop(key, default, kwargs):
    if key in kwargs:
        return kwargs.pop(key)
    else:
        return default

def rolling_window_benchmark(data, train=0.8, **kwargs):
    resample = __pop('resample', None, kwargs)
    output = __pop('output', None, kwargs)

    if resample:
        data = sampling.resample_data(data, resample)

    train_data, test_data = sampling.train_test_split(data, train)

    methods = __pop('methods', None, kwargs)
    orders = __pop("orders", [1, 2, 3], kwargs)
    steps_ahead = __pop('steps_ahead', [1], kwargs)

    for method in methods:
        for order in orders:
            for step in steps_ahead:
                m = method()

                if isinstance(m, fts.FTS):
                    partitioners = __pop("partitioners", [Grid.GridPartitioner], kwargs)
                    partitions = __pop("partitions", [10], kwargs)
                    for partitioner in partitioners:
                        for partition in partitions:
                            data_train_fs = partitioner(data=train_data, npart=partition)
                            m.partitioner = data_train_fs

                # medir tempo de treinamento
                m.fit(train_data, **kwargs)

                # medir tempo de forecast
                yhat = m.predict()
                #_start = time.time()

                # implementar metricas de avaliacao
                _rmse = Measures.rmse(test_data[output].iloc[order:], yhat[:-step])
                print("RMSE: ", _rmse)
                #_end = time.time()

                #TODO:
                # - testar com hofts
                # - implementar barra de progresso
                # - implementar registro de resultados


def prequential_evaluation(method, df, input, output, order, step, train_size, window_size=1, fading_factor=1):
    test_end = 0
    index = 0
    limit = len(df[output])
    accumulated_error = 0
    error_list = []
    fcst = []
    data_input = df[input].values
    data_output = df[output].values

    while test_end < limit:
        train_start, train_end, test_start, test_end = getDataIndex(index, train_size, window_size, limit)

        if (test_end - test_start) > order:
            print(train_start, train_end, test_start, test_end)
            train_data = data_input[train_start:train_end]
            test_data = data_input[test_start:test_end]
            index = test_start

            method.fit(train_data, order=order)
            y_hat_df = pd.DataFrame(method.predict(test_data), columns=input)
            y_hat = y_hat_df[output].to_numpy()[:, 0]

            fcst.extend(y_hat)
            y = data_output[test_start:test_end]

            error = metrics.calculate_rmse(y, y_hat, order, step)
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