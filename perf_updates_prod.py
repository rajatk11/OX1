from polygon import RESTClient
import time
import secrets_ as secrets
import datetime
import psycopg2
import os
import pytz

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

def update_perf(db_cursor):
    # Step 1: Initialize a dictionary to hold total_val for each stock and variables for portfolio total and SP500 value
    total_vals = {}
    portfolio_total = 0
    sp500_value = 0
    update_time = None

    # Step 2: Get start_capital and booked_pnl for each stock from stock_summary
    db_cursor.execute("SELECT stock, start_capital, booked_pnl FROM stock_summary;")
    for stock, start_capital, booked_pnl in db_cursor.fetchall():
        total_vals[stock] = {'start_capital': start_capital, 'booked_pnl': booked_pnl, 'running_pnl': 0, 'avl_cash': 0, 'posn_unit_size': 0}

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
                    if last_trade_time >= last_updated_time :
                        update_perf(cursor)
                    else : 
                        print('No new trades to update')
                        time.sleep(20)
        else :
            print('Market Closed')
            time.sleep(check_time_to_open())

#run main
if __name__ == '__main__':
    main()