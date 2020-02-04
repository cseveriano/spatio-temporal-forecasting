import pandas as pd
from spatiotemporal.test import methods_space_oahu as ms
from spatiotemporal.util import parameter_tuning
from sklearn.metrics import mean_squared_error
from spatiotemporal.data import loader

## load local dataset
hink_cs_df = loader.load_oahu_hinkelman_days(resampling="1min")
hink_cs_df = hink_cs_df['2010-07-31']

methods = []
#methods.append(("HOFTS", ms.hofts_forecast, ms.hofts_space))
#methods.append(("VAR", ms.var_forecast, ms.var_space))
#methods.append(("MLP", ms.mlp_forecast, ms.mlp_space))
methods.append(("EVOLFTS", ms.evolvingfts_forecast, ms.evolvingfts_space))
#methods.append(("FUZZYCNN", ms.fuzzycnn_forecast, ms.fuzzycnn_space))
#methods.append(("GRANULAR", ms.granular_forecast, ms.granular_space))

train = 0.1
parameter_tuning.run_search(methods, hink_cs_df, train, mean_squared_error, max_evals=100, resample=None)

############### Load Tuning Results ##################

#import pickle
#from hyperopt import space_eval
#best = pickle.load(open("best_HOFTS.pkl", "rb"))
#trials = pickle.load(open("trials_HOFTS.pkl", "rb"))
#print('best: ')
#print(space_eval(hofts_space, best))

######################################################




