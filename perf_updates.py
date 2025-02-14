from polygon import RESTClient
import time
import secrets_ as secrets
import datetime
import psycopg2


def is_market_open(api_key):
    client = RESTClient(api_key)
    tr = client.get_market_status()
    return (tr.exchanges.nyse)


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

def main() :
    api_key = secrets.api_key()

    db_params = {
        'dbname' : 'ox1db',
        'host' : 'database-1.c3kqy0cckdw7.us-east-1.rds.amazonaws.com',
        'user' : 't1_write',
        'password' : os.environ.get('DB_PASSWORD'),
    }

    #get max datetime from perf_history
    with psycopg2.connect(**db_params) as nconn:
        with nconn.cursor() as cursor:
            cursor.execute("SELECT MAX(timestamp) FROM perf_history;")
            last_updated_time = cursor.fetchone()[0]
    

    while True :
        if is_market_open(api_key) == "open" :
            with psycopg2.connect(**db_params) as nconn2:
                with nconn2.cursor() as cursor:
                    cursor.execute("SELECT MAX(timestamp) FROM perf_history;")
                    last_updated_time = cursor.fetchone()[0]
                    #get last executed record from trade_inf where status is E
                    cursor.execute("SELECT MAX(datetime) FROM trade_inf WHERE status = 'E';")
                    last_trade_time = cursor.fetchone()[0]
                    if last_trade_time > last_updated_time :
                        update_perf(cursor)
                    else : 
                        print('No new trades to update')
                        time.sleep(20)
        else :
            print('Market Closed')
            time.sleep(60)

