import pandas as pd

def resample_data(df, frequency):
  return df.resample(frequency, how='mean').bfill()


def train_test_split(df, training_length, mode="sample"):

    if mode == "sample":
        limit = round(len(df.index) * training_length)
        train_df = df.iloc[:limit]
        test_df = df.iloc[limit:]
    elif mode == "daily":
        # number of days for training
        limit = round(len(pd.unique(df.index.date)) * training_length)
        ds = pd.Series(df.index.date)

        training_days = pd.unique(df.index.date)[:limit]
        train_df = df.loc[ds.isin(training_days).values, :]

        testing_days = pd.unique(df.index.date)[limit:]
        test_df = df.loc[ds.isin(testing_days).values, :]

    return train_df, test_df
