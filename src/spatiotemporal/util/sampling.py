import pandas as pd

def resample_data(df, frequency):
  return df.resample(frequency, how='mean').bfill()


def train_test_split(df, training_length):
    # number of days for training
    limit = round(len(pd.unique(df.index.date)) * training_length)
    ds = pd.Series(df.index.date)

    training_days = pd.unique(df.index.date)[:limit]
    train_df = df.loc[ds.isin(training_days).values, :]

    testing_days = pd.unique(df.index.date)[limit:]
    test_df = df.loc[ds.isin(testing_days).values, :]

    return train_df, test_df
