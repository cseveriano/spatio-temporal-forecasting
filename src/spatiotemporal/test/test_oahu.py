import pandas as pd
from pyFTS.models import hofts
from pyFTS.partitioners import Grid, Entropy
from spatiotemporal.util import benchmarks

## load local dataset
hink_cs_df = pd.read_csv("../../../data/processed/NREL/Oahu/hink_cs_df.csv",  parse_dates=['Time'], index_col=0)

hink_cs_df.head()

## run benchmark methods
resample = '10min'
output = "DHHL_3"
methods = [hofts.HighOrderFTS]
orders = [1, 2]
steps_ahead = [1]
partitioners = [Grid.GridPartitioner]
partitions = [10]

benchmarks.rolling_window_benchmark(hink_cs_df, train=0.8, resample=resample, output=output, methods=methods, orders=orders,
                         steps_ahead=steps_ahead, partitioners=partitioners, partitions=partitions)
