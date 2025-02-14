

import math
import pandas as pd
import numpy as np
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import datetime
import sys
#sys.path.append('/home/rajat/PycharmProjects/Tidal/Launch_framework/Setup/live_trader/')
#from agent_architecture import DQN
import structs2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import datetime
import time




from configparser import ConfigParser
import psycopg2
from psycopg2 import extras
from psycopg2 import OperationalError

import secrets_ as secrets
# get data



def run_inference(policy_net, featdf, db_cursor, device):
    #checkpoint = torch.load('/home/rajat/Documents/Tidal/Test_1/Launch_files/setup/architectures/dqn_v004.pth')
    checkpoint = torch.load('/home/ubuntu/Tidal/inventory/dqn_v007_3sep_24_good_plus_5.pth', map_location=device)
    trained_net = checkpoint['model']
    policy_net.load_state_dict(trained_net.state_dict())
    #
    #
    featarr = featdf.to_numpy()
    featindarr = pd.DataFrame(
        [featdf.index.get_level_values(0), featdf.index.get_level_values(1)]).to_numpy().transpose()

    feat1arr = np.reshape(featarr[:, 8:71], (featarr.shape[0], 9, 7), order='C')
    feats1 = torch.tensor(feat1arr, dtype=torch.float32)
    feats2 = torch.tensor(featarr[:, 71:98], dtype=torch.float32)

    # feats3 = torch.tensor([[0,0]], dtype=torch.float32)
    trades_list = []

    for i in range(featindarr.shape[0]):

        query = 'SELECT running_pnl, posn_norm from portfolio where stock = %s'
        
        db_cursor.execute(query, (featindarr[i][0],))
        #print('Problem stock : ', featindarr[i][0])
        for row in db_cursor.fetchall():
            print(('ROW : ', row))
            run_pnl, pos_norm = row

        #get the latest stok price from the db
        table_name = featindarr[i][0].lower()  + '1minute' # Ensure this is safe to use
        query = 'SELECT close FROM {0} WHERE datetime = (SELECT MAX(datetime) FROM {0});'.format(table_name)
        db_cursor.execute(query)
        stock_price =  db_cursor.fetchone()[0]




        feats3 = torch.tensor([[pos_norm, (run_pnl*100)/stock_price ]], dtype=torch.float32)

        state1 = feats1[i].unsqueeze_(0)
        state1 = state1.unsqueeze_(0)
        state2 = feats2[i].unsqueeze_(0)
        #test_state = [torch.tensor(state1, dtype=torch.float32, device=device),
        #              torch.tensor(state2, dtype=torch.float32, device=device),
        #              torch.tensor(feats3, dtype=torch.float32, device=device)]

        test_state = [state1.clone().detach().to(dtype=torch.float32, device=device),
                      state2.clone().detach().to(dtype=torch.float32, device=device),
                      feats3.clone().detach().to(dtype=torch.float32, device=device)]



        # print(test_state)
        with torch.no_grad():
            act_ = policy_net(test_state).max(1)[1].view(1, 1)
            if act_.item() == 2:
                act_val = -1
            else:
                act_val = act_.item()
            if abs(act_val + pos_norm) > 1:
                act_val = 0
            trades_list.append(act_val)

       

    # Portfolio_evals.update_portfolio()
    trades_arr = np.array(trades_list)
    #print(trades_arr.shape)
    trades_arr = np.reshape(trades_arr, (trades_arr.shape[0], 1))
    featindarr = np.append(featindarr, trades_arr, axis=1)
    # print(featindarr)
    return featindarr

def connect_with_retry(db_params, retries=5, delay=5):
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

    query = 'SELECT max(datetime) from trade_inf where stock=any(%s)'
    db_cursor.execute(query, (tickers,))

    for row in db_cursor.fetchall():
        last_update = row[0]


    device = "cpu"
    print(device)

    #nnet = DQN
    policy_net = structs2.DQN_t2().to(device)

    feat_file = '/home/ubuntu/Tidal/WIP/temp/net_o.pkl'
    featdf = pd.read_pickle(feat_file)
    tmst = featdf.index.get_level_values(1)[-1]

    #print(tmst)
    #print(last_update)
    if tmst <= last_update:
        print('feature file data not later than last updated')
        db_cursor.close()
        nconn.close()
        return

    featdf = featdf[featdf.index.get_level_values(1).isin([tmst])]
    trades_df = run_inference(policy_net, featdf, db_cursor, device)

    trades_df[:, [0, 1]] = trades_df[:, [1, 0]]
    trades_list = trades_df.tolist()
    for row in trades_list:
        row.append('NE')

    query = "INSERT INTO %s VALUES %%s ON CONFLICT (datetime, stock) DO UPDATE SET datetime = EXCLUDED.datetime, \
              trade = EXCLUDED.trade, status = EXCLUDED.status"
    tabname = 'trade_inf'
    #print(query % (tabname))
    extras.execute_values(db_cursor, query % tabname, trades_list)
    db_cursor.execute("commit;")
    db_cursor.close()
    nconn.close()

    return