import json
import os
from datetime import datetime, timedelta


from xtquant import xtdatacenter as xtdc
xtdc.set_token('4982fbc49f4557b5fcb1927172a8ec4f29e134fa')
xtdc.init()

from xtquant import xtdata


def filter_stocks(config_path: str = "core/download_config.json"):
    """
    Filter stocks based on listing date (> 3 years) and daily average turnover (> 100 million RMB).
    Update the download_config.json with the filtered list.
    Uses xtquant for data.
    """
    if xtdata is None:
        print("Error: xtquant is required but not installed.")
        return

    print("Starting stock filtering using xtquant...")
    
    # 1. Get all A-share stocks
    sector = '沪深A股'
    # Ensure sector info is available (optional download)
    # xtdata.download_sector_data() 
    
    try:
        stock_list = xtdata.get_stock_list_in_sector(sector)
    except Exception as e:
        print(f"Error fetching stock list: {e}")
        return

    print(f"Total stocks in {sector}: {len(stock_list)}")
    
    if not stock_list:
        print("Error: No stocks found. Make sure xtquant is connected/configured correctly.")
        return

    # 2. Filter by listing date (> 3 years)
    print("Filtering by listing date (> 3 years)...")
    today = datetime.now()
    cutoff_date_str = (today - timedelta(days=3*365)).strftime("%Y%m%d")
    
    long_listed_stocks = []
    
    for code in stock_list:
        detail = xtdata.get_instrument_detail(code)
        # detail is a dict, usually has 'OpenDate' (int like 20201010) or 'opendate'
        if detail:
            # Handle potential case sensitivity or missing keys
            open_date = detail.get('OpenDate') or detail.get('opendate')
            if open_date:
                if str(open_date) < cutoff_date_str:
                    long_listed_stocks.append(code)
        else:
            # If no detail, skip
            pass
            
    print(f"Stocks listed > 3 years: {len(long_listed_stocks)}")

    if not long_listed_stocks:
        print("No stocks passed the listing date filter.")
        return

    # 3. Filter by Average Turnover (> 100 million RMB)
    print("Filtering by average turnover (> 100M RMB)...")
    
    # Download data for the last ~40 days to ensure we have enough trading days (20)
    start_date_download = (today - timedelta(days=40)).strftime("%Y%m%d")
    end_date_download = today.strftime("%Y%m%d")
    
    print(f"Downloading history data from {start_date_download} to {end_date_download}...")
    xtdata.download_history_data(long_listed_stocks, period='1d', start_time=start_date_download, end_time=end_date_download)
    
    print("Calculating average turnover...")
    final_stocks = []
    THRESHOLD = 100_000_000 # 100 Million RMB
    
    # Fetch data
    # xtdata.get_market_data_ex returns {stock_code: dataframe}
    # Fields: time, open, high, low, close, volume, amount, ...
    data_map = xtdata.get_market_data_ex(
        stock_list=long_listed_stocks, 
        period='1d', 
        start_time=start_date_download, 
        end_time=end_date_download,
        count=30 # Try to get up to 30 records in range
    )
    
    for code, df in data_map.items():
        if df.empty:
            continue
            
        # Ensure sorted by date
        df = df.sort_index(ascending=True)
        
        # Take last 20 rows
        last_20 = df.tail(20)
        
        if len(last_20) < 10: # Allow some missing data, but need a representative sample
            continue
            
        if 'amount' not in df.columns:
            continue
            
        avg_turnover = last_20['amount'].mean()
        
        if avg_turnover > THRESHOLD:
            final_stocks.append(code)

    print(f"Stocks matching all criteria: {len(final_stocks)}")
    
    # 4. Update download_config.json
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Creating new.")
        # Load existing content if possible or defaults? 
        # But we are updating, so we should try to preserve other keys if it existed but path was wrong? 
        # No, if not exists, just make new.
        config = {}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    new_downloads = []
    
    # Download range: last 1 year from today (standard)
    dl_start_date = (today - timedelta(days=365)).strftime("%Y%m%d")
    dl_end_date = "latest"
    
    for code in final_stocks:
        # xtquant code: "000001.SZ"
        # vnpy config expects symbol: "000001.SZ" (as per example) and separate exchange
        
        if "." not in code:
            continue
            
        symbol_code, suffix = code.split(".")
        
        exchange = ""
        if suffix == "SZ":
            exchange = "SZSE"
        elif suffix == "SH":
            exchange = "SSE"
        elif suffix == "BJ":
            exchange = "BSE"
        else:
            continue
            
        new_downloads.append({
            "symbol": code, # Keep the full symbol as per existing config example
            "exchange": exchange,
            "start_date": dl_start_date,
            "end_date": dl_end_date,
            "interval": "d"
        })
        
    config["downloads"] = new_downloads
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"Successfully updated {config_path} with {len(new_downloads)} stocks.")

if __name__ == "__main__":
    filter_stocks()
