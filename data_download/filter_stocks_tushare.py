import json
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import os

def filter_stocks(config_path: str = "data_download/download_config.json"):
    """
    Filter stocks based on listing date (> 3 years) and daily average turnover (> 100 million RMB).
    Update the download_config.json with the filtered list.
    """
    # 1. Load config to get token
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    token = config.get("tushare_token", "")
    if not token:
        print("Error: tushare_token not found in config.")
        return

    print("Initializing Tushare...")
    ts.set_token(token)
    pro = ts.pro_api()

    # 2. Get Stock Basic Info
    print("Fetching stock basic info...")
    # list_status='L' means listed. fields: ts_code, symbol, name, list_date, market
    df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date,market')
    
    # 3. Filter by listing date (> 3 years)
    print(f"Total listed stocks: {len(df_basic)}")
    today = datetime.now()
    cutoff_date = (today - timedelta(days=3*365)).strftime("%Y%m%d")
    
    # list_date is string 'YYYYMMDD'
    df_filtered_time = df_basic[df_basic['list_date'] < cutoff_date]
    print(f"Stocks listed > 3 years: {len(df_filtered_time)}")
    
    # 4. Filter by Average Turnover (> 100 million RMB)
    # Fetch daily data for the last 20 trading days
    
    trade_days = []
    try:
        # Get trade cal
        df_cal = pro.trade_cal(exchange='', start_date=(today - timedelta(days=60)).strftime("%Y%m%d"), end_date=today.strftime("%Y%m%d"))
        trade_days = df_cal[df_cal['is_open'] == 1]['cal_date'].values[-20:] # Last 20 trading days
        print(f"Using trade calendar. Selected {len(trade_days)} days.")
    except Exception as e:
        print(f"Warning: Failed to fetch trade calendar ({e}). Falling back to checking recent days manually.")
    
    print(f"Calculating average turnover...")
    
    turnover_sum = {}
    count_map = {}
    days_processed = 0
    days_to_check = []
    
    if len(trade_days) > 0:
        days_to_check = trade_days
    else:
        # Fallback: check last 60 days backwards, stop when we get 20 valid days
        for i in range(1, 61):
            d = (today - timedelta(days=i)).strftime("%Y%m%d")
            days_to_check.append(d)
        # We will iterate and stop if we find enough data
    
    valid_days_count = 0
    target_valid_days = 20
    
    for date in days_to_check:
        if valid_days_count >= target_valid_days and len(trade_days) == 0:
            break
            
        print(f"Fetching daily data for {date}...", end='\r')
        try:
            # daily has 'amount' (turnover in thousands RMB)
            df_daily = pro.daily(trade_date=date, fields='ts_code,amount')
            
            if df_daily is None or df_daily.empty:
                continue
                
            valid_days_count += 1
            
            for _, row in df_daily.iterrows():
                ts_code = row['ts_code']
                amount = row['amount']
                
                if pd.isna(amount):
                    continue
                
                if ts_code not in turnover_sum:
                    turnover_sum[ts_code] = 0.0
                    count_map[ts_code] = 0
                
                turnover_sum[ts_code] += amount
                count_map[ts_code] += 1
                
        except Exception as e:
            print(f"\nError fetching data for {date}: {e}")
            if "权限" in str(e):
                print("Permission error detected. Aborting turnover filter.")
                break
                
    print(f"\nDaily data fetch complete. Used data from {valid_days_count} trading days.")

    # Calculate average and filter
    target_stocks = []
    # threshold = 100,000 (thousand RMB) -> 100 million RMB
    THRESHOLD = 100000 
    
    # Only consider stocks that passed the time filter
    valid_ts_codes = set(df_filtered_time['ts_code'].values)
    
    if valid_days_count > 0:
        for ts_code in valid_ts_codes:
            if ts_code in turnover_sum and count_map[ts_code] > 0:
                avg_amount = turnover_sum[ts_code] / count_map[ts_code]
                if avg_amount > THRESHOLD:
                    target_stocks.append(ts_code)
        print(f"Stocks with Avg Turnover > 100M: {len(target_stocks)}")
    else:
        print("No turnover data collected. Skipping turnover filter.")
        # If turnover filter fails, maybe we shouldn't update the list? 
        # Or just return the time-filtered list?
        # User requirement was AND condition. If we can't check turnover, we fail safe?
        # Let's assume we return empty or just time filtered? 
        # Safest is to not destroy the config if we failed completely.
        if valid_days_count == 0:
            print("Failed to filter by turnover. Config will NOT be updated.")
            return

    # 5. Update Config
    # Map tushare code to vnpy symbol and exchange
    new_downloads = []
    # Default range: last 1 year from today
    start_date = (today - timedelta(days=365*5)).strftime("%Y%m%d")
    end_date = "latest"
    
    for ts_code in target_stocks:
        code, suffix = ts_code.split(".")
        exchange_str = ""
        if suffix == "SZ":
            exchange_str = "SZSE"
        elif suffix == "SH":
            exchange_str = "SSE"
        elif suffix == "BJ":
            exchange_str = "BSE" 
        else:
            continue
            
        new_downloads.append({
            "symbol": ts_code,
            "exchange": exchange_str,
            "start_date": start_date,
            "end_date": end_date,
            "interval": "d"
        })
        
    config["downloads"] = new_downloads
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        
    print(f"Updated {config_path} with {len(new_downloads)} stocks.")

if __name__ == "__main__":
    filter_stocks()
