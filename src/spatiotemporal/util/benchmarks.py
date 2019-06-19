import pandas as pd
from pyFTS.benchmarks import Measures
from pyFTS.common import fts
from pyFTS.partitioners import Grid, Entropy, Util as pUtil
from spatiotemporal.util import sampling

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