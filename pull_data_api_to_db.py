
import numpy as np
import pandas as pd
import os
from psycopg2 import extras
import polygon
import sys
import time
import datetime
from configparser import ConfigParser
import psycopg2
import secrets_ as secrets
import pytz
from psycopg2 import OperationalError

def date_slicer(start_date, freq) :
    stdate = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    
    edate = datetime.datetime.now()
    
    intervals = []
    curr = stdate
    
    freqv = str(freq[0]) + freq[1]
    if freqv == '1minute' : window = 20
    elif freqv == '10minute' : window = 200
    elif freqv == '1hour' : window = 12000
    else : window = 100000
    
    while curr <= edate :
        #print(curr)
        #print('HEre')
        c_end = min(datetime.datetime.now(), curr + datetime.timedelta(days = window))
        #print(c_end)
        c_end = min(c_end, edate)
        d1 = str(curr.year) + '-' + "{:02d}".format(curr.month) + '-' + "{:02d}".format(curr.day)
        d2 = str(c_end.year) + '-' + "{:02d}".format(c_end.month) + '-' + "{:02d}".format(c_end.day)
        intervals.append([d1,d2])
        curr = c_end + datetime.timedelta(days = 1)
    return intervals


def pull_data(freq, db_params, stocks) :


    
    api_key = secrets.api_key()
    nconn = connect_with_retry(db_params)
    db_cursor = nconn.cursor()
    symbols = stocks
    #symbols.append('SPY')
    #symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'DIS', 'INTC', 'PYPL', 'CRM', 'IBM', 'SPY']
    #symbols = ['AAPL', 'MSFT']
    #intervals = date_slicer(start_date, end_date, freq)
    #print(symbols)
    
    for sym in symbols :
        print('Processing ' + sym)
        tabname = sym+str(freq[0])+freq[1]
        query = "SELECT MAX (datetime) from %s;"
        #print(query % tabname)
        db_cursor.execute(query % tabname)
        for v in db_cursor.fetchall() :
            #print(v[0])
            pass
        sdate = datetime.datetime.strftime(v[0], '%Y-%m-%d')
        #print(sdate)
        intervals = date_slicer(sdate, freq)
        
        newlist = []
        
        client = polygon.RESTClient(api_key)
        for interval in intervals :
            #print('Processing ' + sym + ' : ' +  interval[0])
            #logging.info('Processing ' + sym + ' : ' +  interval[0])
            try :
                resp = client.list_aggs(ticker = sym, multiplier = freq[0], timespan = freq[1], \
                                        from_ = interval[0], to = interval[1], limit=500000)
                #assert resp[0] == sym
                for rec in resp :
                    try :
                        timest = (datetime.datetime.fromtimestamp(rec.timestamp/1000))

                        timest = datetime.datetime.fromtimestamp(rec.timestamp / 1000, datetime.timezone.utc)
                        bst = pytz.timezone('Europe/London')
                        timest = timest.astimezone(bst)

                        newlist.append((timest, rec.open, rec.high, rec.low, rec.close, rec.volume, rec.vwap,rec.transactions))
                        
                    except :
                        print('ERROR!!!')
                        #logging.error(f'Error for {sym} at {interval[0]}', exc_info=True)
            except :
                print('ERROR   !!')
                #logging.error(f'API call FAILED for {sym} at {interval[0]}', exc_info=True)

            #time.sleep(12)  # to comply with free API restrictions; comment if subscribed
            #logging.info('Download successful for : ', sym)

                        
        
        query = "INSERT INTO %s VALUES %%s ON CONFLICT (datetime) DO NOTHING"
        #print(query % (tabname))
        extras.execute_values(db_cursor, query % tabname, newlist)

    db_cursor.execute("commit;")
    db_cursor.close()
    nconn.close()
    return

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

def main_run(db_params, stocks) :

    stocks_ = stocks.copy()
    stocks_.append('SPY')
    pull_data([1, 'minute'], db_params, stocks_)
    print('Returning from pull_data_api_to_db')
    return
    
#if __name__ == "__main__" : main()