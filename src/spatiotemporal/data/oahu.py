import pandas as pd

def load():
    df = pd.read_csv('https://query.data.world/s/76ohtd4zd6a6fhiwwe742y23fiplgk')

    # drop unused columns
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.drop(['Time', 'Ioh', 'DH1T', 'AP6T', 'AP3', 'AP2.dif', 'AP2.dir'], axis=1, inplace=True)

    # create corrected index
    ind = pd.date_range(start='2010-03-18 00:00:00', end='2011-11-01 00:00:00', closed='left', freq='10s')
    ts = pd.DataFrame(index=ind)
    df['Time'] = ts.between_time("05:00:00", "20:00:00").index
    df.set_index('Time', inplace=True)

    # filter range of interest
    return df["2010-04-01":"2011-10-31"]


# create clear sky index dataframe
def get_clear_sky_index(cs,irr):
  csi = []
  for c,i in zip(cs,irr):
    if c:
      csi.append(i/c)
    else:
      csi.append(0)
  return csi

def load_cs_index_df(df):
    cs_index_df = pd.DataFrame(index=df.index)
    for col in df.columns[2:]:
        cs_index_df[col] = get_clear_sky_index(df['Ics'], df[col])
    return cs_index_df

# get Hinkelman days
def get_hinkelman_days(df):
  hink_days = [212, 213, 214, 215, 216, 217, 233, 241, 248, 249, 250, 264, 300]
  ds = pd.Series(df.index.dayofyear.values)
  return df.loc[[x and y for x, y in zip(ds.isin(hink_days).values, (df.index.year.values == 2010))], :]