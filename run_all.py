

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


def check_time_to_open():
    """
    Checks if it's a weekday after 4pm or a weekend, and returns seconds until next weekday 6am.
    
    Returns:
        int: Seconds until next weekday 6am if conditions are met, otherwise 0
    """
    # Get current datetime in New York time zone
    ny_timezone = pytz.timezone('America/New_York')
    now = datetime.datetime.now(ny_timezone)
    
    # Check if it's a weekday (0 = Monday, 6 = Sunday)
    is_weekday = now.weekday() < 5
    
    # Check if time is after 4pm (16:00)
    is_after_4pm = now.hour >= 16
    
    # If it's a weekday after 4pm or a weekend, calculate time to next opening
    if (is_weekday and is_after_4pm) or not is_weekday:
        # Find the next weekday
        days_ahead = 1
        next_day = now + datetime.timedelta(days=days_ahead)
        
        # Keep adding days until we reach a weekday (Monday-Friday)
        while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
            days_ahead += 1
            next_day = now + datetime.timedelta(days=days_ahead)
        
        # Set time to 6am in New York timezone
        next_opening = ny_timezone.localize(datetime.datetime(
            next_day.year, 
            next_day.month, 
            next_day.day, 
            6, 0, 0  # Hour=6, Minute=0, Second=0
        ))
        
        # Calculate seconds difference
        time_difference = next_opening - now
        return int(time_difference.total_seconds())
    
    # If it's a weekday before 4pm, return 0 (already open)
    return 60
    

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
            time.sleep(check_time_to_open())

    


if __name__ == "__main__": main()



