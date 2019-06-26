from hyperopt import hp
import pandas as pd
import numpy as np
from pyFTS.models import hofts
from pyFTS.partitioners import Grid, Entropy

############# High Order FTS ##############
hofts_space = {'partitioner': hp.choice('partitioner', [Grid.GridPartitioner, Entropy.EntropyPartitioner]),
        'npartitions': hp.choice('npartitions', [10, 50,100]),
        'order': hp.choice('order', [1,2]),
        'input': hp.choice('input', ['DH3']),
        'output': hp.choice('output', ['DH3'])}

def hofts_forecast(train_df, test_df, params):
    _partitioner = params['partitioner']
    _npartitions = params['npartitions']
    _order = params['order']
    _input = params['input']

    fuzzy_sets = _partitioner(data=train_df[_input].values, npart=_npartitions)
    model = hofts.HighOrderFTS(order=_order)

    model.fit(train_df[_input].values, order=_order, partitioner=fuzzy_sets)
    forecast = model.predict(test_df[_input].values)

    return forecast

############# High Order FTS ##############

############# Vector Auto Regressive ##############
from statsmodels.tsa.api import VAR, DynamicVAR

var_space = {
        'order': hp.choice('order', [1,2, 4, 8]),
        'input': hp.choice('input', [['DH3', 'DH4','DH5','DH10','DH11','DH9','DH2', 'DH6','DH7','DH8']]),
        'output': hp.choice('output', ['DH3'])}

def var_forecast(train_df, test_df, params):
    _order = params['order']
    _input = list(params['input'])
    _output = params['output']
    _step = 1

    model = VAR(train_df[_input].values)
    results = model.fit(maxlags=_order)
    lag_order = results.k_ar
    params['order'] = lag_order

    forecast = []
    for i in np.arange(0,len(test_df)-lag_order,_step):
        forecast.extend(results.forecast(test_df[_input].values[i:i+lag_order],_step))

    forecast_df = pd.DataFrame(columns=test_df[_input].columns, data=forecast)
    return forecast_df[_output].values
############# Vector Auto Regressive ##############

############# MultiLayer Perceptron ##############
from keras.models import Sequential
from keras.layers import Dense

mlp_space = {
        'order': hp.choice('order', [1,2, 4, 8]),
        'input': hp.choice('input', [['DH3', 'DH4','DH5','DH10','DH11','DH9','DH2', 'DH6','DH7','DH8']]),
        'output': hp.choice('output', ['DH3']),
        'neurons': hp.choice('neurons', [[8], [64, 32, 8], [8,4]]),
        'epochs': hp.choice('epochs', [50, 100, 150]) }

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def mlp_forecast(train_df, test_df, params):
    _input = list(params['input'])
    _nlags = params['order']
    _neurons = params['neurons']
    _epochs = params['epochs']

    nfeat = len(_input)
    nsteps = 1
    nobs = _nlags * nfeat

    train_reshaped_df = series_to_supervised(train_df[_input], n_in=_nlags, n_out=nsteps)
    train_X, train_Y = train_reshaped_df.iloc[:, :nobs].values, train_reshaped_df.iloc[:, -nfeat].values

    test_reshaped_df = series_to_supervised(test_df[_input], n_in=_nlags, n_out=nsteps)
    test_X, test_Y = test_reshaped_df.iloc[:, :nobs].values, test_reshaped_df.iloc[:, -nfeat].values

    # design network
    model = Sequential()

    for n in _neurons:
        model.add(Dense(n, activation='relu', input_dim=train_X.shape[1]))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit network
    history = model.fit(train_X, train_Y, epochs=_epochs, batch_size=72, verbose=False, shuffle=False)

    forecast = model.predict(test_X)

    return forecast

############# MultiLayer Perceptron ##############


############# Clustered Multivariate FTS ##############

from spatiotemporal.models.clusteredmvfts.fts import cmvhofts
from spatiotemporal.models.clusteredmvfts.partitioner import EvolvingClusteringPartitioner

cmvfts_space = {
        'variance_limit': hp.choice('variance_limit', [0.1, 0.001, 0.0001]),
        't_norm': hp.choice('t_norm', ['threshold', 'nonzero']),
        'defuzzy': hp.choice('defuzzy', ['weighted', 'mean']),
        'order': hp.choice('order', [1,2, 4, 8]),
        'input': hp.choice('input', [['DH3', 'DH4','DH5','DH10','DH11','DH9','DH2', 'DH6','DH7','DH8']]),
        'output': hp.choice('output', ['DH3'])}


def cmvfts_forecast(train_df, test_df, params):

    _variance_limit = params['variance_limit']
    _t_norm = params['t_norm']
    _defuzzy = params['defuzzy']
    _order = params['order']
    _input = list(params['input'])
    _output = params['output']

    fuzzy_sets = EvolvingClusteringPartitioner.EvolvingClusteringPartitioner(data=train_df[_input],
                                                                             variance_limit=_variance_limit, debug=False)
    model = cmvhofts.ClusteredMultivariateHighOrderFTS(t_norm=_t_norm, defuzzy=_defuzzy)
    model.fit(train_df[_input].values, order=_order, partitioner=fuzzy_sets, verbose=False)
    forecast = model.predict(test_df[_input].values)

    forecast_df = pd.DataFrame(data=forecast, columns=test_df[_input].columns)
    return forecast_df[_output].values[:-1]

############# Clustered Multivariate FTS ##############


############# Fuzzy CNN ##############

###### CNN FUNCTIONS ###########
from fts2image import FuzzyImageCNN


###### OPTIMIZATION ROUTINES ###########
fuzzycnn_space = {
        'input': hp.choice('input', [['DH3', 'DH4', 'DH5', 'DH10', 'DH11', 'DH9', 'DH2', 'DH6', 'DH7', 'DH8']]),
        'output': hp.choice('output', ['DH3']),
        'npartitions': hp.choice('npartitions', [100, 150]),
        'order': hp.choice('order', [4, 96,144]),
        'epochs': hp.choice('epochs', [30, 50, 100]),
        'conv_layers' : hp.choice('conv_layers', list(np.arange(2,4))),
        'filters': hp.choice('filters',  [2, 4, 8, 32]),
        'kernel_size': hp.choice('kernel_size', list(np.arange(2,4))),
        'pooling_size': hp.choice('pooling_size', list(np.arange(2,4))),
        'dense_layer_neurons': hp.choice('dense_layer_neurons', [[8], [64, 32, 8], [8,4]]),
        'dropout': hp.choice('dropout', list(np.arange(0.2, 0.5, 0.1))),
        'batch_size':hp.choice('batch_size', [100,200])}

def fuzzycnn_forecast(train_df, test_df, params):
    _input = list(params['input'])
    _npartitions = params['npartitions']
    _order = params['order']
    _conv_layers = params['conv_layers']
    _filters = params['filters']
    _kernel_size = params['kernel_size']
    _pooling_size = params['pooling_size']
    _dense_layer_neurons = params['dense_layer_neurons']
    _dropout = params['dropout']
    _batch_size = params['batch_size']
    _epochs = params['epochs']

    fuzzy_sets = Grid.GridPartitioner(data=train_df[_input].values, npart=_npartitions).sets
    model = FuzzyImageCNN.FuzzyImageCNN(fuzzy_sets, nlags=_order, steps=1,
            conv_layers = _conv_layers,
            filters = _filters, kernel_size = _kernel_size,
            pooling_size = _pooling_size, dense_layer_neurons = _dense_layer_neurons, dropout=_dropout, debug=False)

    model.fit(train_df[_input], batch_size=_batch_size, epochs=_epochs)

    forecast = model.predict(test_df[_input])
    return [f[0] for f in forecast]

############# Fuzzy CNN ##############

