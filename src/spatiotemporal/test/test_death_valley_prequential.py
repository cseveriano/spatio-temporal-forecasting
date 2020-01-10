import pandas as pd
import numpy as np
from fbem.FBeM import FBeM
from fbem.utils import *
from spatiotemporal.models.clusteredmvfts.fts import evolvingclusterfts
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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


def getDataIndex(index, window_size, limit):
    train_start = index

    train_end = index + window_size

    test_start = train_end
    test_end = min(test_start + window_size, limit)

    return train_start, train_end, test_start, test_end

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
    df = pd.DataFrame(data)
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
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


xls = pd.ExcelFile("../../../data/processed/FBEM/DeathValleyAvg.xls")
sheetx = xls.parse(0)

_order = 2
_step = 1

# Preparing x
x = []
for index, row in sheetx.iterrows():
    x = x + row[1:].tolist()


fbi = FBeM()
fbi.debug = True
to_normalize = 1

# Normalize data
if to_normalize:
    fbi.rho = 0.3
    min_v = min(x)
    max_v = max(x)
    x_norm = normalize(array=x, min=min_v, max=max_v)
else:
    min_v = min(x)
    max_v = max(x)
    fbi.rho = 0.2 * (max_v - min_v)

df = series_to_supervised(x_norm, n_in=_order, n_out=_step)


########## PREQUENTIAL EVALUATION #################################

window_size = 200
test_end = 0
index = 0
limit = len(df.index)
accumulated_error = 0
fading_factor = 1
evolving_error_list = []
fbem_error_list = []
evolving_fcst = []
fbem_fcst = []

############# CREATE METHODS ############################

evolving = evolvingclusterfts.EvolvingClusterFTS(t_norm='nonzero', defuzzy='weighted', variance_limit=0.001)
data_input = df.iloc[:,:_order].values
data_output = df.iloc[:,-1].values

#########################################################

while test_end < limit:
    train_start, train_end, test_start, test_end = getDataIndex(index, window_size, limit)

    if (test_end - test_start) > _order:
        print(train_start, train_end, test_start, test_end)
        train_data = data_input[train_start:train_end]
        test_data = data_input[test_start:test_end]
        index = test_start

        #### EVOLVING FTS ####
        evolving.fit(train_data, order=_order, num_batches=None)
        y_hat_df = pd.DataFrame(evolving.predict(test_data), columns=df.columns[:_order])
        y_hat = y_hat_df.iloc[:, -1].values

        evolving_fcst.extend(y_hat)
        y = data_output[test_start:test_end]

        error = calculate_rmse(y, y_hat, _order, _step)
        accumulated_error += fading_factor * error
        evolving_error_list.append(error)

#        plt.plot(y, 'k-', label="Expected output")
#        plt.plot(y_hat, 'b-', label="Predicted output")

        ######################

        ##### FBEM ###########

        ## Ler base treinamento ##
        y_train = data_output[train_start:train_end]
        for sample, obs in zip(train_data, y_train):
            fbi.learn(x=sample.tolist(), y=[obs])
        ## Ler base teste ##
        fbem_yhat = []
        for sample, obs in zip(test_data, y):
            fbi.learn(x=sample.tolist(), y=[obs])
            fbem_yhat.append(fbi.P[test_start:test_end])
        fbem_pred = fbem_yhat[-1][(_order-1):]
        fbem_fcst.extend(fbem_pred)
        ## Salvar output de teste ##
        error = calculate_rmse(y, fbem_pred, _order, _step)
        accumulated_error += fading_factor * error
        fbem_error_list.append(error)

        plt.plot(y, 'k-', label="Expected output")
        plt.plot(y_hat, 'b-', label="Predicted output")

        ######################



x = []
y = []

evolving_yhat = []
evolving_error_list = []
persistence_yhat = []
persistence_error_list = []

for i in range(0, axis_1):
    x = []
    y = []
    for j in range(0, axis_2):
        x.append(xavg[j][i])

    y.append(yavg[i])

    fbi.learn(x=x, y=y)

    de = np.empty((0,2),float)
    for i in x:
        de = np.append(de, np.array([[i] * 2]), axis=0)
    de = np.append(de, np.array([y * 2]), axis=0)
    evolving.fit(de, order=n, num_batches=None)
    y_hat = evolving.predict(x)
    y_hat = y_hat[-1]
    evolving_yhat.append(y_hat)

    part = y - y_hat
    part = power(part, 2)
    part = sum_(part)
    part = np.sqrt(part / (fbi.h + 1))
    evolving_error_list.append(part)

    persistence_yhat.append(x[-1])
    part = y - x[-1]
    part = power(part, 2)
    part = sum_(part)
    part = np.sqrt(part / (fbi.h + 1))
    persistence_error_list.append(part)

## Comparar RMSE
print("FBeM - Average RMSE: ", fbi.rmse[len(fbi.rmse) - 1])
print("Evolving FTS - Average RMSE: ", evolving_error_list[len(fbi.rmse) - 1])
print("Persistence - Average RMSE: ", persistence_error_list[len(fbi.rmse) - 1])
fbi.file.close()

## Comparar graficos
limit = 200
fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(yavg[:limit], 'k-', label="Expected output")
axs[0].plot(fbi.P[:limit], 'b-', label="Predicted output")
axs[0].set_title('FBeM')
fig.suptitle('Death Valley Dataset', fontsize=16)

axs[1].plot(yavg[:limit], 'k-', label="Expected output")
axs[1].plot(evolving_yhat[:limit], 'b-', label="Predicted output")
axs[1].set_title('Evolving FTS')

axs[1].plot(yavg[:limit], 'k-', label="Expected output")
axs[1].plot(persistence_yhat[:limit], 'b-', label="Predicted output")
axs[1].set_title('Persistence')

plt.show()