import pandas as pd
from spatiotemporal.util import common

## load local dataset
hink_cs_df = pd.read_csv("../../../data/processed/NREL/Oahu/hink_cs_df.csv")

hink_cs_df.head()

## run colab processing lines
hink_cs_df = common.resample_data(hink_cs_df, "10min")
hink_train_cs_df, hink_test_cs_df = common.train_test_split(hink_cs_df, 0.5)


## create fit and forecast methods

