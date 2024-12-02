import pull_data_api_to_db

import datetime
import pytz
from getpass import getpass

import secrets_ as secrets
import psycopg2
import time

#tickers = ['AAPL','MSFT','AMZN','META','TSLA','NVDA','GOOGL','DIS','INTC','ADBE','CRM','PYPL','AMD', 'IBM', 'ORCL', 'CSCO', 'AVGO', 'QCOM']
tickers = ['SPY']
db_params = {
    'dbname' : 'ox1db',
    'host' : 'database-1.c3kqy0cckdw7.us-east-1.rds.amazonaws.com',
    'user' : 't1_write',
    'password' : getpass("Enter db password for t1_write: "), }

pull_data_api_to_db.main_run(db_params, tickers)

print('Data pulled from API and stored in database')