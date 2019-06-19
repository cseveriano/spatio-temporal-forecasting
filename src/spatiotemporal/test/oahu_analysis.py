import pandas as pd
from pyFTS.models import hofts
from pyFTS.partitioners import Grid, Entropy
from spatiotemporal.util import benchmarks

## load local dataset
hink_raw_df = pd.read_csv("../../../data/processed/NREL/Oahu/hink_raw_df.csv",  parse_dates=['Time'], index_col=0)

hink_raw_df.head()
