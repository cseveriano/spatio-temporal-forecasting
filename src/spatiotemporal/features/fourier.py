import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.fft import fft, ifft


def remove_periodic(X, df_index, detrending=True, model='additive', frequency_threshold=0.1e12):
    rad = np.array(X)

    if detrending:
        det_rad = rad - np.average(rad)
    else:
        det_rad = rad

    det_rad_fft = fft(det_rad)

    # Get the power spectrum
    rad_ps = [np.abs(rd) ** 2 for rd in det_rad_fft]

    clean_rad_fft = [det_rad_fft[i] if rad_ps[i] > frequency_threshold else 0
                     for i in range(len(det_rad_fft))]

    rad_series_clean = ifft(clean_rad_fft)
    rad_series_clean = [value.real for value in rad_series_clean]

    if detrending:
        rad_trends = rad_series_clean + np.average(rad)
    else:
        rad_trends = rad_series_clean

    rad_clean_ts = pd.Series(rad_trends, index=df_index)

    # rad_clean_ts[(rad_clean_ts.index.hour < 6) | (rad_clean_ts.index.hour > 20)] = 0
    residual = rad - rad_clean_ts.values
    clean = rad_clean_ts.values

    return residual, clean