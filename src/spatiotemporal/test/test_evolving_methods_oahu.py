from spatiotemporal.data import loader
from spatiotemporal.models.clusteredmvfts.fts import evolvingclusterfts
from pyFTS.models.multivariate import granular
from spatiotemporal.util import benchmarks
from spatiotemporal.models.benchmarks.fbem import FBeM
from spatiotemporal.models.benchmarks.var import var
from spatiotemporal.models.benchmarks.granularfts import granularfts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Load Dataset
1 - Death Valley
2 - Oahu Hinkelman days
3 - Weather Sao Paulo
'''
#loader.load_weather_sao_paulo()
#loader.load_death_valley(2,1)
data = loader.load_oahu_hinkelman_days(resampling="15min")
#data = loader.load_oahu_raw_qualification()
#interval = ((data.index >= '2010-06') & (data.index < '2011-06'))
#data = data.loc[interval]

## Single day experiment
#data = data["2010-07-31"]

## Six days (50%) experiment
data = data["2010-07-31":"2010-08-05"]

'''
Define parameters
- Input columns
- Output column
- Initial training size
- Window size
- Resampling
- Order
- Step
'''
#_ini_train_size = len(data["2010-07-31"]) # an entire day of initial training
_ini_train_size = 100
_window_size = 100
_order = 2
_step = 1
_input = ['DH4', 'DH5', 'DH6']
_output = ['DH4']
#_input = ['DHHL_4', 'DHHL_5', 'DHHL_6']
#_output = ['DHHL_4']


'''
Adapt Forecasting Methods
- Evolving FTS
- FBeM
- VAR
- Persistence
'''

## FIG
_npartitions = 100
_alpha_cut = 0.3
_knn = 3

#fig_model = granularfts.GranularFTS(data=data[_input], membership='triangular',
#                                    npartitions=_npartitions, alpha_cut=_alpha_cut, knn=_knn, order=_order)
#fig_accumulated_error, fig_error_list, fig_fcst = benchmarks.prequential_evaluation(fig_model, data, _input, _output, _order, _step, _ini_train_size, _window_size)

## Evolving FTS
evolving_model = evolvingclusterfts.EvolvingClusterFTS(defuzzy='weighted', membership_threshold=0.6, variance_limit=0.001)
evol_accumulated_error, evol_error_list, evol_fcst = benchmarks.prequential_evaluation(evolving_model, data, _input, _output, _order, _step, _ini_train_size, _window_size)

#plt.plot(data[_output[0]][_ini_train_size+_window_size+_order:].values, 'k-', label="Expected output")
#plt.plot(evol_fcst, 'b-', label="Predicted output")

## FBEM
fbi_model = FBeM.FBeM()
fbi_model.debug = True
fbi_model.n = _order
fbem_accumulated_error, fbem_error_list, fbem_fcst = benchmarks.prequential_evaluation(fbi_model, data, _output, _output, _order, _step, _ini_train_size, _window_size)

## VAR
var_model = var.VectorAutoregressive()
var_accumulated_error, var_error_list, var_fcst = benchmarks.prequential_evaluation(var_model, data, _input, _output, _order, _step, _ini_train_size, _window_size)


results_df = pd.DataFrame({'EvolvingFTS': evol_error_list, 'FBeM': fbem_error_list, 'VAR': var_error_list})
ax = results_df.plot(figsize=(18, 6), yticks=np.arange(0, 1.1, 0.1))
ax.set(xlabel='Window', ylabel='Adj. Rand Index')
#fig = ax.get_figure()
#fig.savefig(path_images + exp_id + "_prequential.png")
