

import pull_data_api_to_db as step1
import data_featurizer as step2
import trader_agent as step3
import trade_exec as step4
import datetime
import pytz
from getpass import getpass
from polygon import RESTClient

import secrets_ as secrets
from agent_architecture import DQN
import psycopg2
import time
import multiprocessing as mp
import os



def is_market_open(api_key):
    client = RESTClient(api_key)
    tr = client.get_market_status()
    return (tr.exchanges.nyse)

#print(is_market_open('0_2MWuIiIwpeIBwEpgQDaQlYSJhMkaQw') != "open")

def set_status(db_params, status, status_code) :
    nconn2 = psycopg2.connect(**db_params)
    cursor = nconn2.cursor()
    query = "UPDATE algo_status SET curr_status = %s, curr_status_code = %s"
    cursor.execute(query, (status, status_code))
    nconn2.commit()
    cursor.close()
    nconn2.close()


def main() :
    mp.set_start_method('spawn')
    api_key = secrets.api_key()

    db_params = {
        'dbname' : 'ox1db',
        'host' : 'database-1.c3kqy0cckdw7.us-east-1.rds.amazonaws.com',
        'user' : 't1_write',
        'password' : os.environ.get('DB_PASSWORD'),
    }
    nconn2 = psycopg2.connect(**db_params)
    cursor = nconn2.cursor()
    query = "SELECT max(datetime) from trade_inf"
    cursor.execute(query)
    last_update = cursor.fetchone()[0]
    cursor.close()
    nconn2.close()

    tickers = ['AAPL','MSFT','AMZN','META','TSLA','NVDA','GOOGL','DIS','INTC','ADBE','CRM','PYPL','AMD', 'IBM', 'ORCL', 'CSCO', 'AVGO', 'QCOM']
    #tickers = ['AAPL','MSFT','AMZN','META','TSLA','NVDA','GOOGL','DIS','INTC','ADBE','CRM','PYPL']

    while True :
        if is_market_open(api_key) == "open" :
            print('Start_run')
            set_status(db_params, 'Getting latest market data', 1)
            step1.main_run(db_params, tickers)
            print('STEP 2')
            set_status(db_params, 'Tokenizing : Converting data to feature set', 2)
            step2.main_run(db_params, tickers)
            print('STEP 3')
            set_status(db_params, 'Inferring : Generating trade ides', 3)
            step3.main_run(db_params, tickers)
            print('STEP 4')
            set_status(db_params, 'Executing trades and updating database', 4)
            step4.main_run(db_params)
            
        else :
            set_status(db_params, 'Market Closed', 0)
            print('Market Closed')
            time.sleep(60)

    


if __name__ == "__main__": main()



