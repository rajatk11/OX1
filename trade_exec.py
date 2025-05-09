
import math
import random
from collections import namedtuple, deque
from itertools import count
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import datetime
import decimal
import pytz
from psycopg2 import OperationalError
import secrets_ as secrets

from configparser import ConfigParser
import psycopg2
from psycopg2 import extras

import polygon


def is_market_open():
    client = polygon.RESTClient(secrets.api_key())
    tr = client.get_market_status()
    return (tr.exchanges.nyse)


def get_days_open(stock):
    client = polygon.RESTClient(secrets.api_key())
    tr = client.get_snapshot_ticker("stocks", stock)
    tmst = datetime.datetime.fromtimestamp(tr.min.timestamp / 1000, datetime.timezone.utc)
    bst = pytz.timezone('Europe/London')
    tmst = tmst.astimezone(bst)
    #tmst = pytz.utc.localize(tmst)
    if is_market_open() == "open":
        return (tr.day.close, tmst)
    else:
        return (tr.prev_day.close, tmst)


def get_prev_close(stock):
    client = polygon.RESTClient(secrets.api_key())
    tr = client.get_snapshot_ticker("stocks", stock)
    tmst = datetime.datetime.fromtimestamp(tr.min.timestamp / 1000, datetime.timezone.utc)
    bst = pytz.timezone('Europe/London')
    tmst = tmst.astimezone(bst)
    #tmst = pytz.utc.localize(tmst)
    return (tr.prev_day.close, tmst)


def get_curr_price(stock):
    client = polygon.RESTClient(secrets.api_key())
    tr = client.get_snapshot_ticker("stocks", stock)
    tmst = datetime.datetime.fromtimestamp(tr.min.timestamp / 1000, datetime.timezone.utc)
    bst = pytz.timezone('Europe/London')
    tmst = tmst.astimezone(bst)

    return (tr.min.close, tmst)


def get_quote_bid(stock):
    client = polygon.RESTClient(secrets.api_key())
    tr = client.get_snapshot_ticker("stocks", stock)
    tmst = datetime.datetime.fromtimestamp(tr.min.timestamp / 1000, datetime.timezone.utc)
    bst = pytz.timezone('Europe/London')
    tmst = tmst.astimezone(bst)

    return (tr.min.close, tmst)


def get_quote_ask(stock):
    client = polygon.RESTClient(secrets.api_key())
    tr = client.get_snapshot_ticker("stocks", stock)
    tmst = datetime.datetime.fromtimestamp(tr.min.timestamp / 1000, datetime.timezone.utc)
    bst = pytz.timezone('Europe/London')
    tmst = tmst.astimezone(bst)

    return (tr.min.close, tmst)


def get_trades_to_exec(db_cursor):
    query = "SELECT * from trade_inf where status = 'NE';"
    db_cursor.execute(query)

    trades_list = []
    for row in db_cursor.fetchall():
        trades_list.append(row)

    return trades_list


def exec_trades(trlist, db_cursor):
    execlist = []
    for row in trlist:
        print('Entering', row)
        if row[-3] == 'NE':
            trade_datetime, stock, dirn, _,_, _ = row
        else:
            continue

        if dirn == 1:
            exec_price, exec_time = get_quote_ask(stock)
        elif dirn == -1:
            exec_price, exec_time = get_quote_bid(stock)
        else:
            exec_price, exec_time = get_curr_price(stock)

        query = "UPDATE trade_inf set exec_time = %s, exec_price = %s where stock = %s and datetime = %s;"
        db_cursor.execute(query, (exec_time, exec_price, stock, trade_datetime))
        execlist.append([exec_time, stock, dirn, exec_price])

    return execlist


def update_portfolio_trade(exec_list, db_cursor):
    # in portfolio table, update : posn_norm, posn_val, avl_cash, last_trade_time,
    # posn_'seq'_price, posn_'seq'_tmst, posn_size
    trade_sq_list = []
    # db_cursor = nconn2.cursor()

    for trade in exec_list:
        trade_time, stock, dirn, exec_price = trade
        print('TRADE TIME : ', trade_time)
        print(stock, dirn)

        #print(stock)
        query = 'SELECT posn_norm, posn_val, avl_cash from portfolio where stock = %s;'
        db_cursor.execute(query, (stock,))

        for val in db_cursor.fetchall():
            posn_norm, posn_val, avl_cash= val

        if abs(posn_norm + dirn) > 1 or dirn == 0:
            pass
        else:

            if posn_norm == 0 :  # to determine if we're adding to position or reducing

                posn_norm += dirn
                exec_price = decimal.Decimal(exec_price)
                posn_unit_size = avl_cash // exec_price
                posn_val = decimal.Decimal(dirn * exec_price * posn_unit_size)

                # posn_add = 1
                posn_seq_price_colname = 'posn_1_price'
                posn_seq_tmst_colname = 'posn_1_tmst'

                pos_seq_price_val = exec_price
                posn_seq_tmst_val = trade_time.strftime("%Y-%m-%d %H:%M:%S")

                avl_cash -= decimal.Decimal(abs(posn_val))

                try:
                    query2 = 'UPDATE portfolio SET posn_norm = {pn}, posn_val = {pv}, avl_cash = {ac}, \
                          last_trade_time = %s, posn_unit_size = {ps} WHERE stock = %s;'.format(pn=posn_norm, \
                                                                                                pv=posn_val,
                                                                                                ac=avl_cash,
                                                                                                ps=posn_unit_size)
                    db_cursor.execute(query2, (trade_time, stock))
                    query3 = 'UPDATE portfolio SET {pspc} = {pspv}, {pstc} = %s where stock = %s;'.format(
                        pspc=posn_seq_price_colname, \
                        pspv=pos_seq_price_val, pstc=posn_seq_tmst_colname)
                    db_cursor.execute(query3, (posn_seq_tmst_val, stock))
                    # nconn2.commit()
                except psycopg2.Error as err:
                    print('ERROR!! @ update portfolio when posn_add')
                    print(stock, dirn)
                    print(err)
                    # nconn2.rollback()


            else:
                #18 July. 24 - start here ------------------------------------------------------------------------------------------------
                #Alter the code to allow only one position per stock
                # posn_add = 0

                print('Stock closing : ', stock)

                query = 'SELECT avl_cash, posn_unit_size from portfolio where stock = %s;'
                db_cursor.execute(query, (stock,))

                for val in db_cursor.fetchall():
                    avl_cash, posn_unit_size = val

                posn_norm = 0
                posn_val = 0

                #decimal.Decimal(dirn * exec_price * posn_unit_size))



                posn_seq_price_colname = 'posn_1_price'
                posn_seq_tmst_colname = 'posn_1_tmst'

                query1 = 'SELECT {pspc}, {pstc} from portfolio where stock = %s;'.format(pspc=posn_seq_price_colname,
                                                                                         pstc=posn_seq_tmst_colname)
                db_cursor.execute(query1, (stock,))
                for val in db_cursor.fetchall():
                    trade_in_price = val[0]
                    trade_in_tmst = val[1]
                trade_sq_list.append([stock, dirn, exec_price, trade_time, trade_in_price, trade_in_tmst])

                pos_seq_price_val = None
                posn_seq_tmst_val = None
                avl_cash += decimal.Decimal(abs(posn_unit_size * exec_price))


                try:
                    query2 = 'UPDATE portfolio SET posn_norm = {pn}, posn_val = {pv}, avl_cash = {ac}, \
                          last_trade_time = %s, posn_unit_size = {ps} WHERE stock = %s;'.format(pn=posn_norm, \
                                                                                                pv=posn_val,
                                                                                                ac=avl_cash,
                                                                                                ps=posn_unit_size)
                    db_cursor.execute(query2, (trade_time, stock))
                    query3 = 'UPDATE portfolio SET {pspc} = %s, {pstc} = %s where stock = %s;'.format(
                        pspc=posn_seq_price_colname, \
                        pstc=posn_seq_tmst_colname)
                    db_cursor.execute(query3, (pos_seq_price_val, posn_seq_tmst_val, stock))
                    # nconn2.commit()

                except psycopg2.Error as err:
                    print('ERROR!! @ update portfolio when NOT posn_add')

        query4 = 'UPDATE portfolio SET last_updated_time = %s where stock = %s;'
        db_cursor.execute(query4, (trade_time, stock))
        # nconn2.commit()

    return trade_sq_list


def update_stock_summary(tr_up_list, db_cursor):
    for rec in tr_up_list:

        stock, dirn, exec_price, trade_time, trade_in_price, trade_in_tmst = rec
        exec_price = decimal.Decimal(exec_price)
        trade_profit = (exec_price - trade_in_price) * dirn * -1
        trade_durn = (trade_time - trade_in_tmst).total_seconds() // 60
        print('TRADe DURN : ', trade_durn)
        query = "SELECT * from stock_summary where stock = %s"
        db_cursor.execute(query, (stock,))

        for row in db_cursor.fetchall():
            _, longs_ct, shorts_ct, avg_hold_period, _, booked_pnl = row

        #get posn_unit_size from portfolio
        query2 = "SELECT posn_unit_size from portfolio where stock = %s"
        db_cursor.execute(query2, (stock,))
        for row in db_cursor.fetchone():
            posn_unit_size = row

        if dirn == 1:
            shorts_ct += 1
        else:
            longs_ct += 1

        avg_hold_period += (trade_durn - avg_hold_period) / (longs_ct + shorts_ct)

        booked_pnl += (trade_profit * posn_unit_size)

        upquery = "UPDATE stock_summary set longs_count = {lct}, shorts_count = {sct}, avg_hold_period_mins = {avp}, \
                   booked_pnl = {bpnl} where stock = %s".format(lct=longs_ct, sct=shorts_ct, avp=avg_hold_period, \
                                                                bpnl=booked_pnl)
        db_cursor.execute(upquery, (stock,))

    return


def update_portfolio_stats(db_cursor):
    #intervals = []
    #t1 = datetime.datetime.now()

    query = "SELECT stock, posn_norm, posn_val, avl_cash, posn_1_price, posn_unit_size from portfolio"
    db_cursor.execute(query)
    #intervals.append((datetime.datetime.now() - t1))

    for row in db_cursor.fetchall():
        stock, posn_norm, pon_val, avl_cash, p1, posn_unit_size = row


        curr_price = decimal.Decimal(get_curr_price(stock)[0])
       # if len(intervals) == 1:
        #    intervals.append(datetime.datetime.now() - t1)
        #else:
         #   intervals[1] += (datetime.datetime.now() - t1)

        if posn_norm > 0:
            posn_dirn = 1
        elif posn_norm < 0:
            posn_dirn = -1
        else:
            posn_dirn = 0

        if p1 is None:
            p1 = 0
       # if p2 is None:
        #    p2 = 0
        #if p3 is None:
         #   p3 = 0


        running_pnl = (curr_price - abs(p1)) * posn_dirn

        open_today = get_days_open(stock)[0]
        open_today = decimal.Decimal(open_today)

        prev_day_close = get_prev_close(stock)[0]
        prev_day_close = decimal.Decimal(prev_day_close)

        days_gain_val = (abs(curr_price)- prev_day_close) * posn_dirn

        #if len(intervals) == 2:
        #   intervals.append(datetime.datetime.now() - t1)
       # else:
        #    intervals[2] += (datetime.datetime.now() - t1)
        #t1 = datetime.datetime.now()

        query2 = "UPDATE portfolio set running_pnl = {rpl}, open_today = {opt}, days_gain_val = {dgv} where \
                  stock = %s".format(rpl=running_pnl, opt=open_today, dgv=days_gain_val)
        db_cursor.execute(query2, (stock,))

        #if len(intervals) == 3:
         #   intervals.append(datetime.datetime.now() - t1)
        #else:
        #    intervals[3] += (datetime.datetime.now() - t1)

    #print(intervals)
    return


def change_trade_status(db_cursor):

    query = "update trade_inf set status = %s where status = %s;"
    db_cursor.execute(query, ('E', 'NE'))
    return


def update_perf(db_cursor):
    # Step 1: Initialize a dictionary to hold total_val for each stock and variables for portfolio total and SP500 value
    total_vals = {}
    portfolio_total = 0
    sp500_value = 0
    update_time = None

    # Step 2: Get start_capital and booked_pnl for each stock from stock_summary
    db_cursor.execute("SELECT stock, start_capital, booked_pnl FROM stock_summary;")
    for stock, start_capital, booked_pnl in db_cursor.fetchall():
        total_vals[stock] = {'start_capital': start_capital, 'booked_pnl': booked_pnl, 'running_pnl': 0, 'avl_cash': 0, posn_unit_size: 0}

    # Step 3: Get running_pnl and avl_cash for each stock from portfolio table
    db_cursor.execute("SELECT stock, running_pnl, avl_cash, posn_unit_size FROM portfolio;")
    for stock, running_pnl, avl_cash, posn_unit_size in db_cursor.fetchall():
        if stock in total_vals:
            total_vals[stock]['running_pnl'] = running_pnl
            total_vals[stock]['avl_cash'] = avl_cash
            total_vals[stock]['posn_unit_size'] = posn_unit_size

    # Step 4 & 5: Calculate total_val for each stock and find the latest update_time
    for stock in total_vals:
        vals = total_vals[stock]
        total_val = vals['start_capital'] + vals['booked_pnl'] + (vals['running_pnl'] * vals['posn_unit_size'])
        total_vals[stock]['total_val'] = total_val
        portfolio_total += total_val

    # Get the latest update_time from the portfolio table
    db_cursor.execute("SELECT MAX(last_updated_time) FROM portfolio;")
    update_time_result = db_cursor.fetchone()
    update_time = update_time_result[0] if update_time_result else None

    # Step 6: Insert into perf_history for each stock
    for stock, vals in total_vals.items():
        db_cursor.execute("INSERT INTO perf_history (stock, timestamp, value) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                          (stock, update_time, vals['total_val']))

    # Step 7: Insert into perf_history for the complete portfolio
    db_cursor.execute("INSERT INTO perf_history (stock, timestamp, value) VALUES ('portfolio', %s, %s) ON CONFLICT DO NOTHING;",
                      (update_time, portfolio_total))

    # Step 8: Insert into perf_history for SP500
    db_cursor.execute("SELECT close FROM spy1minute ORDER BY datetime DESC LIMIT 1;")
    sp500_value = db_cursor.fetchone()[0] if db_cursor.rowcount > 0 else 0
    db_cursor.execute("INSERT INTO perf_history (stock, timestamp, value)  VALUES ('SP500', %s, %s) ON CONFLICT DO NOTHING;",
                      (update_time, sp500_value))


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

def main_run(db_params):
    nconn = connect_with_retry(db_params)
    db_cursor = nconn.cursor()

    trlist = get_trades_to_exec(db_cursor)
    print('TR list done', trlist)
    exec_list = exec_trades (trlist, db_cursor)
    print('HERE we go......' , exec_list)
    tr_up_lst = update_portfolio_trade(exec_list, db_cursor)
    print(tr_up_lst)
    update_stock_summary(tr_up_lst, db_cursor)
    print('3')
    update_portfolio_stats(db_cursor)
    print('4')
    change_trade_status(db_cursor)
    #update_perf(db_cursor)
    db_cursor.execute('commit;')
    db_cursor.close()
    nconn.close()
    # exec_list = exec_trades(trlist)
    return

