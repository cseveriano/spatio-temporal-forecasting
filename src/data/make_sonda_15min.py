# -*- coding: utf-8 -*-
import pandas as pd
from pandas import read_csv
import glob
import sys

pattern = sys.argv[1]

input_filepath = '../../data/raw/SONDA'
output_filepath = '../../data/processed/SONDA/'+pattern+'-15min.csv'


header_file = '../../data/raw/SONDA/ED_header_new.csv'
hd = read_csv(header_file, sep = ';')

# mount new header
cl = hd.columns[4:8].tolist()
cl.extend(hd.columns[10:])
cl.insert(0,"date")
new_header = pd.DataFrame(columns = cl)
new_header.to_csv(output_filepath, sep = ';', index=False)

for file in glob.glob(input_filepath + '/' + pattern + '*'):
    print (file + '\n')

    df = read_csv(file, header=0, sep = ';')
    df.columns = hd.columns


    year = df.year
    day_of_year = df.day
    minute_of_day = df['min']


    date_col = pd.to_datetime(year * 1000 + day_of_year, format='%Y%j') + pd.TimedeltaIndex(minute_of_day, unit='m')

    data = pd.concat([df.iloc[:, df.columns.get_loc('glo_avg'):df.columns.get_loc('par_avg')], df.iloc[:, df.columns.get_loc('tp_sfc'):]], axis=1)
    data.index = date_col
    data = data.resample('15min').mean()
    ## salvar arquivo output
    data.to_csv(output_filepath, mode='a', header=False, sep = ';', na_rep='NaN')

