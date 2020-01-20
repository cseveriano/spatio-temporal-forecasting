from sklearn.cluster import MeanShift, estimate_bandwidth

import pandas as pd
import glob
from pathlib import Path

def load_data_nrel(path, resampling=None):
    ## some resampling options: 'H' - hourly, '15min' - 15 minutes, 'M' - montlhy
    ## more options at:
    ## http://benalexkeen.com/resampling-time-series-data-with-pandas/
    allFiles = glob.iglob(path + "/**/*.txt", recursive=True)
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        #print("Reading: ",file_)
        df = pd.read_csv(file_,index_col="datetime",parse_dates=['datetime'], header=0, sep=",")
        if frame.columns is None :
            frame.columns = df.columns
        list_.append(df)
    frame = pd.concat(list_)
    if resampling is not None:
        frame = frame.resample(resampling).mean()
    frame = frame.fillna(method='ffill')

    frame.columns = ['DHHL_3', 'DHHL_4', 'DHHL_5', 'DHHL_10', 'DHHL_11', 'DHHL_9', 'DHHL_2', 'DHHL_1', 'DHHL_1_Tilt',
                  'AP_6', 'AP_6_Tilt', 'AP_1', 'AP_3', 'AP_5', 'AP_4', 'AP_7', 'DHHL_6', 'DHHL_7', 'DHHL_8']

    return frame


def create_spatio_temporal_data_oahu(oahu_df):
    lat = [21.31236,21.31303,21.31357,21.31183,21.31042,21.31268,21.31451,21.31533,21.30812,21.31276,21.31281,21.30983,21.31141,21.31478,21.31179,21.31418,21.31034]
    lon = [-158.08463,-158.08505,-158.08424,-158.08554,-158.0853,-158.08688,-158.08534,-158.087,-158.07935,-158.08389,-158.08163,-158.08249,-158.07947,-158.07785,-158.08678,-158.08685,-158.08675]
    additional_info = pd.DataFrame({'station': oahu_df.columns, 'latitude': lat, 'longitude': lon })
    ll = []
    for ind, row in oahu_df.iterrows():
        for col in oahu_df.columns:
            lat = additional_info[(additional_info.station == col)].latitude.values[0]
            lon = additional_info[(additional_info.station == col)].longitude.values[0]
            irradiance = row[col]
            ll.append([lat, lon, irradiance])

    return pd.DataFrame(columns=['latitude','longitude','irradiance'], data=ll)

def load_oahu_dataset(start_date = "2010-04-01", end_date = "2011-10-31"):
    """
    Dataset used in
    "Impact of network layout and time resolution on spatio-temporal solar forecasting" - Amaro e Silva, R. C. Brito, M. - Solar Energy 2018

    :param start_date: time series start date in dd-mm-yyyy
    :param end_date: time series end date in dd-mm-yyyy
    :return: dataset in dataframe
    """

    # read raw dataset
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
    df = df[start_date:end_date]
    return df

# create clear sky index dataframe
def get_clear_sky_index(cs,irr):
  csi = []
  for c,i in zip(cs,irr):
    if c:
      csi.append(i/c)
    else:
      csi.append(0)
  return csi

def load_oahu_dataset_clear_sky(start_date = "2010-04-01", end_date = "2011-10-31"):
    raw_df = load_oahu_dataset(start_date, end_date)
    cs_index_df = pd.DataFrame(index=raw_df.index)

    for col in raw_df.columns[2:]:
        cs_index_df[col] = get_clear_sky_index(raw_df['Ics'], raw_df[col])

    return cs_index_df


def normalize_data(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)


def load_oahu_hinkelman_days(normalize=True, zenith_angle=80):
    hink_raw_df = pd.read_csv('https://query.data.world/s/4is4okebsk5vi2ok5utiwyhmdlko7n', parse_dates=['Time'],index_col=0)

    if normalize:
        hink_raw_df = normalize_data(hink_raw_df)

    if zenith_angle is not None:
        zenith_angle_index = hink_raw_df.zen < 80
        hink_raw_df = hink_raw_df[zenith_angle_index]

    return hink_raw_df

def load_oahu_hinkelman_days_clear_sky(normalize=True, zenith_angle=80):
    hink_cs_df = pd.read_csv('https://query.data.world/s/six7uxsdqen6s47qf7m7mzs4r3iz2d',  parse_dates=['Time'], index_col=0)

    if normalize:
        hink_cs_df = normalize_data(hink_cs_df)

    if zenith_angle is not None:
        zenith_angle_index = hink_cs_df.zen < 80
        hink_cs_df = hink_cs_df[zenith_angle_index]

    return hink_cs_df

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.

    Code adapted from Machine Learning Mastery:
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_death_valley(_order, _step, normalize=True):
    xls = pd.ExcelFile(Path(__file__).resolve().parents[3] / "data/processed/FBEM/DeathValleyAvg.xls")
    sheetx = xls.parse(0)

    # Preparing x
    x = []
    for index, row in sheetx.iterrows():
        x = x + row[1:].tolist()

    if normalize:
        min_v = min(x)
        max_v = max(x)
        x = [(xk - min_v) / (max_v - min_v) for xk in x]

    df = series_to_supervised(x, n_in=_order, n_out=_step)

    return df


def load_weather_sao_paulo(normalize=True):
    """
    Data originally extracted from INMET - Instituto Nacional de Meteorologia (inmet.gov.br)

    The dataset contains monthly data of air humidity, cloudiness, rainfall, maximum temperature, minimum temperature, and mean temperature obtained from four brazilian weather stations: Manaus, Sao Paulo, Natal, and Porto Alegre. Data were collected from January of 1990 to December of 2015.
    Number of samples : 312. Number of attributes: 6.

    The dataset was used in:
    Soares, E.; Costa Jr., P.; Costa, B.; Leite, D. "Ensemble of Evolving Data Clouds and Fuzzy Models for Weather Time Series Prediction." Applied Soft Computing - Elsevier, xx-x, 2018.

    :return: dataset
    """
    df = pd.read_excel(Path(__file__).resolve().parents[3] / "data/processed/Weather/SaoPaulo.xlsx", parse_dates=['Date'], index_col=0)

    if normalize:
        df = normalize_data(df)

    return df
