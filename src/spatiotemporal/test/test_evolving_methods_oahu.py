from spatiotemporal.data import loader
from spatiotemporal.models.clusteredmvfts.fts import evolvingclusterfts
from spatiotemporal.util import benchmarks
from spatiotemporal.models.benchmarks.fbem import FBeM
from spatiotemporal.models.benchmarks.var import var

'''
Load Dataset
1 - Death Valley
2 - Oahu Hinkelman days
3 - Weather Sao Paulo
'''
#loader.load_weather_sao_paulo()
#loader.load_death_valley(2,1)
data = loader.load_oahu_hinkelman_days()

## Filter zenith angle
zenith_angle_index = data.zen < 80
data = data[zenith_angle_index]

## Single day experiment
data = data["2010-07-31"]

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
_ini_train_size = 100
_window_size = 100
_order = 2
_step = 1
_input = ['DH4', 'DH5', 'DH6']
_output = ['DH4']
_step = 1

'''
Adapt Forecasting Methods
- Evolving FTS
- FBeM
- VAR
- Persistence
'''

## Evolving FTS
#evolving_model = evolvingclusterfts.EvolvingClusterFTS(t_norm='nonzero', defuzzy='weighted', debug=False)
#evol_accumulated_error, evol_error_list, evol_fcst = benchmarks.prequential_evaluation(evolving_model, data, _input, _output, _order, _step, _ini_train_size, _window_size)

## FBEM
#fbi_model = FBeM.FBeM()
#fbi_model.debug = True
#fbi_model.n = _order
#fbem_accumulated_error, fbem_error_list, fbem_fcst = benchmarks.prequential_evaluation(fbi_model, data, _output, _output, _order, _step, _ini_train_size, _window_size)

## VAR
var_model = var.VectorAutoregressive()
var_accumulated_error, var_error_list, var_fcst = benchmarks.prequential_evaluation(var_model, data, _input, _output, _order, _step, _ini_train_size, _window_size)




