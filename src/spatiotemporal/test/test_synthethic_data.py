import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns


from pyFTS.common import Util

from pyFTS.data import artificial

mu_local = 5
sigma_local = 0.25
mu_drift = 3
sigma_drift = 0.5
deflen = 200
totlen = deflen * 10
order = 5

signals = {}


def mavg(l, order=2):
    ret = []  # l[:order]
    for k in np.arange(order, len(l)):
        ret.append(np.nanmean(l[k - order:k]))

    return ret


signal = artificial.SignalEmulator().stationary_gaussian(mu_local, sigma_local, length=deflen, it=10).run()

signals['Stationary signal'] = mavg(signal, order)

signal = artificial.SignalEmulator().stationary_gaussian(mu_local, sigma_local, length=deflen,
                                                         it=10).blip().blip().run()

signals['Stationary signal with blip'] = mavg(signal, order)

signal = artificial.SignalEmulator() \
    .stationary_gaussian(mu_local, sigma_local, length=deflen // 2, it=10) \
    .stationary_gaussian(mu_local, sigma_drift, length=deflen // 2, it=10, additive=False) \
    .blip().blip() \
    .run()

signals['Sudden Variance'] = mavg(signal, order)

signal = artificial.SignalEmulator() \
    .stationary_gaussian(mu_local, sigma_local, length=deflen // 2, it=10) \
    .stationary_gaussian(mu_drift, sigma_local, length=deflen // 2, it=10, additive=False) \
    .blip().blip() \
    .run()

signals['Sudden Mean'] = mavg(signal, order)

signal = artificial.SignalEmulator() \
    .stationary_gaussian(mu_local, sigma_local, length=deflen // 2, it=10) \
    .stationary_gaussian(mu_drift, sigma_drift, length=deflen // 2, it=10, additive=False) \
    .blip().blip() \
    .run()

signals['Sudden Mean & Variance'] = mavg(signal, order)

signal = artificial.SignalEmulator() \
    .stationary_gaussian(mu_local, sigma_local, length=deflen, it=10) \
    .incremental_gaussian(0.1, 0, length=totlen // 2, start=totlen // 2) \
    .blip().blip() \
    .run()

signals['Incremental Mean'] = mavg(signal, order)

signal = artificial.SignalEmulator() \
    .stationary_gaussian(mu_local, sigma_local, length=deflen, it=10) \
    .incremental_gaussian(0., 0.1, length=totlen // 2, start=totlen // 2) \
    .blip().blip() \
    .run()

signals['Incremental Variance'] = signal

signal = artificial.SignalEmulator() \
    .stationary_gaussian(mu_local, sigma_local, length=deflen, it=10) \
    .incremental_gaussian(0.02, 0.01, length=totlen // 2, start=totlen // 2) \
    .blip().blip() \
    .run()

signals['Incremental Mean & Variance'] = mavg(signal, order)


from spatiotemporal.models.clusteredmvfts.fts import evolvingclusterfts
from pyFTS.models.multivariate import granular
from spatiotemporal.util import benchmarks
from spatiotemporal.models.benchmarks.fbem import FBeM
from pyFTS.benchmarks import Measures
from spatiotemporal.data import loader

fig, ax = plt.subplots(nrows=18, ncols=1, figsize=[15,25])

rows = []
_order = 2

for row, key in enumerate(signals.keys()):
    print('Processing dataset: ', key)
    df = loader.series_to_supervised(signals[key], n_in=_order, n_out=1)
    data_input = df.iloc[:,:_order].values
    data_output = df.iloc[:,-1].values

    l = len(df.index)
    limit = l//2
    train = data_input[:limit]
    test = data_input[limit:]

    ax[row].plot(test, label="Original")
    ax[row].set_title(key)

    evolving_model = evolvingclusterfts.EvolvingClusterFTS(defuzzy='weighted', membership_threshold=0.6, variance_limit=0.001)
    evolving_model.fit(train, order=_order)
    y_hat_df = pd.DataFrame(evolving_model.predict(test))
    forecasts = y_hat_df.iloc[:, -1].values
    ax[row].plot(forecasts, label="EvolvingFTS")
    _rmse = Measures.rmse(data_output[limit+_order:], forecasts[:-1])
    data = [key, "EvolvingFTS", _rmse]
    rows.append(data)

    fbem_model = FBeM.FBeM()
    fbem_model.n = _order
    fbem_model.fit(train, order=_order)
    forecasts = fbem_model.predict(test)
    ax[row].plot(forecasts, label="FBeM")
    _rmse = Measures.rmse(data_output[limit+_order:], forecasts[:-1])
    data = [key, "FBeM", _rmse]
    rows.append(data)

    handles, labels = ax[row].get_legend_handles_labels()
    lgd = ax[row].legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

plt.tight_layout()

pd.DataFrame(rows,columns=["Dataset","Model","RMSE"]).sort_values(by=["RMSE"])
