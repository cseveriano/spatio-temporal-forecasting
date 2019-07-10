import os
import pandas as pd
from spatiotemporal.test import methods_space_oahu as ms
from spatiotemporal.util import experiments as ex

## load local dataset
hink_cs_df = pd.read_csv("../../../data/processed/NREL/Oahu/hink_cs_df.csv",  parse_dates=['Time'], index_col=0)
hink_cs_df = hink_cs_df['2010-07-31']


path = "C:\\Users\\cseve\\Google Drive\\SpatioTemporal_Results\\Forecasting"
os.chdir(path)

train_split = 0.5

# exp_id = "EXP_1_MLP"
# method = ms.mlp_forecast
# space = ms.mlp_space


# exp_id = "EXP_1_GRANULAR"
# method = ms.granular_forecast
# space = ms.granular_space


exp_id = "EXP_1_VAR"
method = ms.var_forecast
space = ms.var_space
#
params = ex.load_best_params(exp_id, space)
params['step'] = 2


# method = ms.persistence_forecast
# params = {'order': 3, 'output': 'DH4', 'step': 2}

#(rmse, smape, u) = ex.forecast_params(hink_cs_df, train_split, method, params, plot=True)
window_size = 100
(rmse, smape, u) = ex.rolling_window_forecast_params(hink_cs_df, train_split, window_size, method, params)


