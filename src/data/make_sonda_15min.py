# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from pandas import read_csv
from pandas import datetime


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def make_sonda_15min(input_filepath, output_filepath):

    #varrer diretorio com arquivos csvs
    #input recebe diretorio com arquivos
    # listar arquivos a partir do nome
    #iniciar for com concatenacao dos arquivos


    header_file = '../../data/raw/SONDA/ED_header_new.csv'
    hd = read_csv(header_file, sep = ';')
    df = read_csv(input_filepath, header=0, sep = ';')
    df.columns = hd.columns


    year = df.year
    day_of_year = df.day
    minute_of_day = df['min']


    date_col = pd.to_datetime(year * 1000 + day_of_year, format='%Y%j') + pd.TimedeltaIndex(minute_of_day, unit='m')

#    print(df)

    data = pd.DataFrame(df.glo_avg)
    data.index = date_col
    data = data.resample('15min').mean()
    ## salvar arquivo output

    ## append data to csv file
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())


    input_filepath = '../../data/raw/SONDA/FLN1311ED.csv'
    output_filepath = '../../data/processed'
    make_sonda_15min(input_filepath, output_filepath)
