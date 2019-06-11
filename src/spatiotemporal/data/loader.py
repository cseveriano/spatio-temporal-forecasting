from sklearn.cluster import MeanShift, estimate_bandwidth

import pandas as pd
import glob

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