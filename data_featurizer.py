from configparser import ConfigParser
import numpy as np
import psycopg2
import pandas as pd
import os
from psycopg2 import extras
import datetime
import sys
sys.path.append('..')
import time
import math
import random

import multiprocessing as mp

import secrets_ as secrets

sys.path.append(secrets.feat_builder_locn())
from psycopg2 import OperationalError
import Features.feat_builder as featlib


def create_feat_set(df):
    builder = featlib.build_features(df)

    # add index

    index_file = secrets.index_file()

    # filter openhours

    builder.filter_openhours()
    builder.add_index_vals(index_file, 'Close', 'sp500')



    # Set 1 : minute by minute day's change data
    builder.get_day_change('Close', 'temp_day_change')
    # print(builder.df.dtypes)
    builder.feat_norm_signs('temp_day_change', window=40000, min_vals=38000, name='day_change_0')
    # builder.df.drop(columns=['temp_day_change'], inplace=True)

    # Add day_change for upto last 6 minutes; for the immediate interval upon opening, add 0 for columns where T-n
    # falls before the open. Eg. day_change_4 will be 0 a 9:32AM
    base_feat_name = 'day_change_'
    for i in range(1, 7):
        builder.df[base_feat_name + str(i)] = ((builder.df.index - builder.df.index.normalize()).seconds >= 34200 + (
                    i * 60)) * \
                                              builder.df['day_change_0'].shift(i)

    builder.df.drop(columns='temp_day_change', inplace=True)

    # Set 2 :
    builder.sma('Close', window=200, minvals=190, name='temp_200_sma')
    builder.df['temp_close_by_sma'] = builder.df['Close'] / builder.df['temp_200_sma'] - 1
    builder.feat_norm_signs('temp_close_by_sma', window=40000, min_vals=39000, name='close_over_200_sma_0')
    builder.df.drop(columns=['temp_close_by_sma'], inplace=True)

    base_feat_name = 'close_over_200_sma_'
    for lags in [20, 50, 100, 200, 800, 2000]:
        tempname = 'temp_200_sma_' + str(lags)
        feat_name = base_feat_name + str(lags)
        builder.df[tempname] = builder.df['Close'] / builder.df['temp_200_sma'].shift(lags) - 1
        builder.feat_norm_signs(tempname, 40000, 39000, feat_name)
        builder.df.drop(columns=[tempname], inplace=True)

    builder.df.drop(columns=['temp_200_sma'], inplace=True)

    # Set 3 :
    builder.sma('Close', window=2000, minvals=1900, name='temp_2000_sma')
    builder.df['temp_close_by_sma'] = builder.df['Close'] / builder.df['temp_2000_sma'] - 1
    builder.feat_norm_signs('temp_close_by_sma', window=40000, min_vals=39000, name='close_over_2000_sma_0')
    builder.df.drop(columns=['temp_close_by_sma'], inplace=True)

    base_feat_name = 'close_over_2000_sma_'
    for lags in [50, 200, 400, 800, 2000, 10000]:
        tempname = 'temp_2000_sma_' + str(lags)
        feat_name = base_feat_name + str(lags)
        builder.df[tempname] = builder.df['Close'] / builder.df['temp_2000_sma'].shift(lags) - 1
        builder.feat_norm_signs(tempname, 40000, 39000, feat_name)
        builder.df.drop(columns=[tempname], inplace=True)

    builder.df.drop(columns=['temp_2000_sma'], inplace=True)

    # Set 4 :
    builder.sma('Close', window=20000, minvals=19000, name='temp_20000_sma')
    builder.df['close_over_20000_sma_0'] = builder.df['Close'] / builder.df['temp_20000_sma'] - 1

    base_feat_name = 'close_over_20000_sma_'
    for lags in [800, 4000, 12000, 20000, 40000, 80000]:
        feat_name = base_feat_name + str(lags)
        builder.df[feat_name] = builder.df['Close'] / builder.df['temp_20000_sma'].shift(lags) - 1
        

    builder.df.drop(columns=['temp_20000_sma'], inplace=True)

    # Set 5 :
    builder.sma('Close', window=80000, minvals=78000, name='temp_80000_sma')
    builder.df['close_over_80000_sma_0'] = builder.df['Close'] / builder.df['temp_80000_sma'] - 1

    base_feat_name = 'close_over_80000_sma_'
    for lags in [800, 4000, 12000, 20000, 40000, 80000]:
        feat_name = base_feat_name + str(lags)
        builder.df[feat_name] = builder.df['Close'] / builder.df['temp_80000_sma'].shift(lags) - 1

    builder.df.drop(columns=['temp_80000_sma'], inplace=True)

    # Set 6 :

    builder.get_cumulative_vol('Vol', 'Cumu_vol')
    builder.get_prev_close('Cumu_vol', 'prev_eod_vol')
    builder.sma('prev_eod_vol', 1200, 1100, 'temp_3d_vol_sma')

    builder.feat_norm_fixed_time('Cumu_vol', 100, 100, 'Vol_cum_day_0')  # 1

    builder.feat_std_norm('prev_eod_vol', 20000, 19000, 'Vol_prev_eod_norm_50d')
    builder.feat_std_norm('prev_eod_vol', 40000, 39000, 'Vol_prev_eod_norm_100d')
    builder.feat_std_norm('prev_eod_vol', 80000, 79000, 'Vol_prev_eod_norm_200d')

    builder.feat_std_norm('temp_3d_vol_sma', 20000, 19000, 'Vol_prev_3d_norm_50d')
    builder.feat_std_norm('temp_3d_vol_sma', 40000, 39000, 'Vol_prev_3d_norm_100d')
    builder.feat_std_norm('temp_3d_vol_sma', 80000, 79000, 'Vol_prev_3d_norm_200d')

    builder.df.drop(columns=['Cumu_vol', 'prev_eod_vol', 'temp_3d_vol_sma'], inplace=True)

    # Set 7
    builder.get_day_change('sp500', 'temp_index_day_change')
    builder.feat_norm_signs('temp_index_day_change', window=40000, min_vals=39000, name='index_day_change_0')
    # builder.df.drop(columns=['temp_day_change'], inplace=True)

    # Add day_change for upto last 6 minutes; for the immediate interval upon opening, add 0 for columns where T-n
    # falls before the open. Eg. day_change_4 will be 0 a 9:32AM
    base_feat_name = 'index_day_change_'
    for i in range(1, 7):
        builder.df[base_feat_name + str(i)] = ((builder.df.index - builder.df.index.normalize()).seconds >= 34200 + \
                                               (i * 60)) * builder.df['index_day_change_0'].shift(i)
    builder.df.drop(columns='temp_index_day_change', inplace=True)

    # Set 8 :
    builder.sma('sp500', window=20000, minvals=19000, name='temp_index_20000_sma')
    builder.df['index_over_20000_sma_0'] = builder.df['sp500'] / builder.df['temp_index_20000_sma'] - 1
    base_feat_name = 'index_over_20000_sma_'
    for lags in [800, 4000, 12000, 20000, 40000, 80000]:
        feat_name = base_feat_name + str(lags)
        builder.df[feat_name] = builder.df['sp500'] / builder.df['temp_index_20000_sma'].shift(lags) - 1
        
    builder.df.drop(columns=['temp_index_20000_sma'], inplace=True)

    # Set 9 :
    builder.sma('sp500', window=80000, minvals=60000, name='temp_index_80000_sma')
    builder.df['index_over_80000_sma_0'] = builder.df['sp500'] / builder.df['temp_index_80000_sma'] - 1
    base_feat_name = 'index_over_80000_sma_'
    for lags in [800, 4000, 12000, 20000, 40000, 80000]:
        feat_name = base_feat_name + str(lags)
        builder.df[feat_name] = builder.df['sp500'] / builder.df['temp_index_80000_sma'].shift(lags) - 1
    builder.df.drop(columns=['temp_index_80000_sma'], inplace=True)

    # 10 :

    builder.get_beta('sp500', 'Close', 1200, 1100, 'beta_3d')
    builder.df['beta_3d'] = np.tanh(builder.df['beta_3d'] - 1)

    builder.get_beta('sp500', 'Close', 4000, 3900, 'beta_10d')
    builder.df['beta_10d'] = np.tanh(builder.df['beta_10d'] - 1)

    builder.get_beta('sp500', 'Close', 20000, 19000, 'beta_50d')
    builder.df['beta_50d'] = np.tanh(builder.df['beta_50d'] - 1)

    builder.get_beta('sp500', 'Close', 80000, 79000, 'beta_200d')
    builder.df['beta_200d'] = np.tanh(builder.df['beta_200d'] - 1)

    # Add time(minute) of the day
    builder.minute_of_day()

    
    builder.df.dropna(inplace=True)

    return


def load_spy(db_cursor):

    starttime = datetime.datetime.now()
    d1 = starttime - datetime.timedelta(days=650)
    d1 = datetime.datetime.strftime(d1, '%Y-%m-%d')
    tabname = 'SPY1minute'
    query = "select * from {tn} where datetime >= %s order by datetime;".format(tn=tabname)

    db_cursor.execute(query, (d1,))
    dat = pd.DataFrame(db_cursor.fetchall())
    cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Vol', 'vwPrice', 'Numbars']
    dat.columns = cols
    dat['Datetime'] = pd.to_datetime(dat['Datetime'], utc=True)
    dat['Datetime'] = pd.to_datetime(dat['Datetime']) \
        .dt.tz_convert('America/New_York')
    dat.dropna(inplace=True)
    dat = dat.astype({
        'Open': 'float64', 'High': 'float64', 'Low': 'float64', 'Close': 'float64',
        'Vol': 'float64', 'vwPrice': 'float64', 'Numbars': 'int64'})
    dat.set_index('Datetime', inplace=True)

    dat.to_pickle('/home/ubuntu/Tidal/WIP/temp/SPY1minute.pkl')
    return


def feat_stock(td):
    ticker, dat = td
    
    #print('Ticker : ', ticker)
    create_feat_set(dat)
    dat['Timestamp'] = dat.index
    dat['Ticker'] = ticker
    dat.set_index(['Ticker', 'Timestamp'], inplace=True)
    #print('Finished ', ticker)

    return dat

def connect_with_retry(db_params, retries=3, delay=3):
    for attempt in range(retries):
        try:
            nconn = psycopg2.connect(**db_params, connect_timeout=10)
            return nconn
        except OperationalError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def main_run(db_params, tickers):

    
    nconn = connect_with_retry(db_params)
    db_cursor = nconn.cursor()
    

    #print('IN 1')
    pool = mp.Pool(processes = 2)
    starttime = datetime.datetime.now()
    d1 = starttime - datetime.timedelta(days=650)
    d1 = datetime.datetime.strftime(d1, '%Y-%m-%d')
    dflist = []
    stlist = []

    for ticker in tickers[:]:
        #print(ticker)
        tabname = ticker + '1minute'
        query = "select * from {tn} where datetime >= %s order by datetime;".format(tn=tabname)

        db_cursor.execute(query, (d1,))
        dat = pd.DataFrame(db_cursor.fetchall())
        cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Vol', 'vwPrice', 'Numbars']
        dat.columns = cols
        dat['Datetime'] = pd.to_datetime(dat['Datetime'], utc=True)
        dat['Datetime'] = pd.to_datetime(dat['Datetime']) \
            .dt.tz_convert('America/New_York')

        dat.dropna(inplace=True)
        dat = dat.astype({'Open': 'float64', 'High': 'float64', 'Low': 'float64', 'Close': 'float64',
                          'Vol': 'float64', 'vwPrice': 'float64', 'Numbars': 'int64'})
        dat.set_index('Datetime', inplace=True)
        # print(dat.dtypes)

        dflist.append([ticker, dat.copy()])


    load_spy(db_cursor)
    #print('Starting multiprocessing pool')
    stlist = pool.map(feat_stock, dflist)
    pool.close()
    pool.join()
    #print('Finished multiprocessing pool')

    preclist = []
    for df in stlist:
        preclist.append(df.iloc[-1:, :].copy())
    del stlist
    sumdf = pd.concat(preclist)
    #print(sumdf)

    sumdf.to_pickle('/home/ubuntu/Tidal/WIP/temp/net_o.pkl')
    db_cursor.close()
    nconn.close()
    
    return

