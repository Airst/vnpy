import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.selector.selector import FundamentalSelector
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.utility import load_json, get_folder_path
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.trader.database import get_database

# Gateway and App
from vnpy_tora import ToraStockGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.engine import StrategyEngine

# Strategy
from core.strategies.multifactor_strategy import MultiFactorStrategy

# Settings
STRATEGY_NAME = "daily_multifactor"
GATEWAY_NAME = "TORASTOCK"
CONNECT_FILE = "connect_torastock.json"
SIGNAL_NAME = "ashare_mlp_signal_v4"

def run_daily_trading():
    """
    Main execution function for daily trading.
    1. Connect to Tora Gateway
    2. Init Portfolio Strategy
    3. Load Market Data (from DB)
    4. Trigger Strategy
    """
    print(f"[{datetime.now()}] Starting Daily Trading Module...")

    # 1. Initialize Engines
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    selector = FundamentalSelector()
    
    main_engine.add_gateway(ToraStockGateway)
    portfolio_engine: StrategyEngine = main_engine.add_app(PortfolioStrategyApp)
    
    # Register custom strategy class
    portfolio_engine.classes["MultiFactorStrategy"] = MultiFactorStrategy
    
    # 2. Connect to Gateway
    print(f"[{datetime.now()}] Connecting to {GATEWAY_NAME}...")
    
    # Try to find connect config
    # 1. Current dir
    # 2. .vntrader dir
    setting = load_json(CONNECT_FILE)
    if not setting:
        # Try .vntrader
        vntrader_path = get_folder_path(".")
        setting = load_json(str(vntrader_path.joinpath(CONNECT_FILE)))
        
    if not setting:
        print(f"Error: Could not find {CONNECT_FILE} in current directory or .vntrader")
        # In a real scenario, we might want to exit, but if the user configured it in the UI, 
        # it should be in .vntrader.
        # Check if we can connect without explicit setting (some gateways load internal config)
        # But Tora usually needs it.
        # We will proceed attempting to connect with empty dict if fails, or warn.
        print("Attempting to connect with empty setting (expecting internal config load)...")
        setting = {}

    main_engine.connect(setting, GATEWAY_NAME)
    
    # Wait for connection (simple sleep, ideal would be checking status)
    # Tora connection might take a few seconds
    # During this time, Gateway queries Trade/Position automatically.
    print(f"[{datetime.now()}] Waiting 10s for Gateway connection and data sync...")
    time.sleep(10)
    print(f"[{datetime.now()}] Gateway Status: {main_engine.get_gateway(GATEWAY_NAME).default_setting}") 

    # 3. Initialize Strategy
    print(f"[{datetime.now()}] Initializing Strategy...")
    portfolio_engine.init_engine() # This loads 'portfolio_strategy_data.json'
    
    # Get or Add Strategy
    portfolio_engine.remove_strategy(STRATEGY_NAME)  # For testing, remove existing to start fresh
    strategy = portfolio_engine.strategies.get(STRATEGY_NAME)
    if not strategy:
        print(f"Strategy {STRATEGY_NAME} not found in persistence. Creating new instance...")
        # Note: We need a dummy vt_symbol list or a full universe.
        # The strategy loads signals to determine what to buy.
        # But for 'vt_symbols' argument, we usually pass the universe.
        # Let's pass an empty list and let logic handle it, OR pass all known symbols if possible.
        # Since MultiFactorStrategy calculates 'target_symbols' from signals, 
        # we can start with empty and let it build up or passing empty is fine if the engine allows.
        # However, PortfolioStrategy usually subscribes to these symbols.
        # Since we are not subscribing (manual feed), empty might be okay, 
        # BUT 'strategy.vt_symbols' is used for position checks.
        
        portfolio_engine.add_strategy(
            class_name="MultiFactorStrategy", 
            strategy_name=STRATEGY_NAME, 
            vt_symbols=[], 
            setting={
                "signal_name": SIGNAL_NAME,
                "max_holdings": 5,
                "capital": 5000000
            }
        )
        strategy = portfolio_engine.strategies[STRATEGY_NAME]
        
    # Manually trigger on_init if not done by init_engine (init_engine usually inits all)
    # If we just added it, add_strategy calls init_strategy.
    if not strategy.inited:
        portfolio_engine.init_strategy(STRATEGY_NAME)
        print(f"[{datetime.now()}] Waiting for strategy initialization...")
        while not strategy.inited:
            time.sleep(1)

    print(f"[{datetime.now()}] Starting Strategy...")
    portfolio_engine.start_strategy(STRATEGY_NAME)
        

    # 4. Prepare Data & Sync Trades
    # Date: Today
    date_range = selector.get_data_range()  # Ensure selector is ready (loads DB)
    today = date_range[1]  # Use max date available in DB
    
    print(f"[{datetime.now()}] Preparing Data for {today}...")
    
    # Identify Symbols
    # A. From Signals (Candidates to Buy)
    today_str = today.strftime("%Y-%m-%d")
    signal_scores = strategy.signal_data.get(today_str, {})
    print(f"Max Date: {max(strategy.signal_data.keys()) if strategy.signal_data else 'None'}")
    candidate_symbols = list(signal_scores.keys())
    
    # B. From Holdings (Candidates to Sell)
    # We need to know what we hold.
    # Strategy maintains 'pos_entry_price' which implies holding.
    held_symbols = list(strategy.pos_entry_price.keys())
    
    all_symbols: Set[str] = set(candidate_symbols) | set(held_symbols)

    # --- Manual Trade Sync (CRITICAL for Daily Execution) ---
    # Since we restart the process daily, the engine lost the OrderID map.
    # We must manually inject today's trades into the strategy to update its state (pos, avg_price).
    print(f"[{datetime.now()}] Syncing today's trades...")
    all_trades = main_engine.get_all_trades()
    sync_count = 0
    for trade in all_trades:
        # Check if this trade belongs to our interested symbols
        # Note: In a shared account, this might pick up manual trades too. 
        # Ideally, filter by order_ref if possible, but map is lost.
        if trade.vt_symbol in all_symbols:
            print(f"  -> Replaying Trade: {trade.vt_symbol} {trade.direction} {trade.volume} @ {trade.price}")
            strategy.update_trade(trade)
            sync_count += 1
    print(f"[{datetime.now()}] Synced {sync_count} trades.")
    
    
    if not all_symbols:
        print(f"No symbols found for today (Signals: {len(candidate_symbols)}, Held: {len(held_symbols)}). Exiting.")
        main_engine.close()
        return

    print(f"[{datetime.now()}] Loading bars for {len(all_symbols)} symbols...")
    
    database = get_database()
    bars_dict: Dict[str, BarData] = {}
    
    for vt_symbol in all_symbols:
        try:
            symbol, exchange_str = vt_symbol.split(".")
            exchange = Exchange(exchange_str)
        except ValueError:
            # Handle symbols like '000001.SZ' -> '000001.SZSE' if needed, 
            # but usually strategy uses proper vt_symbol.
            # If signal has '000001.SZ', we need conversion.
            # MultiFactorStrategy code suggests it uses what's in signal file.
            # Let's assume signal file has valid vt_symbols or handle exception.
            print(f"Warning: Invalid vt_symbol format {vt_symbol}")
            continue

        # Load ONE DAY of data
        # We assume the DB has data for 'today'.
        # Note: load_bar_data interval is inclusive.
        # We load just today.
        bars = database.load_bar_data(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.DAILY,
            start=today,
            end=today
        )
        
        if bars:
            bars_dict[vt_symbol] = bars[0] # Take the daily bar
        else:
            # If no data for today (e.g. suspended or data not updated), we skip
            # print(f"No data for {vt_symbol}")
            pass

    print(f"[{datetime.now()}] Loaded {len(bars_dict)} bars.")
    
    if not bars_dict:
        print("No market data found for today. Aborting strategy trigger.")
        # We might still want to run on_bars with empty to trigger internal logic? 
        # No, usually need data.
        main_engine.close()
        return

    # 5. Trigger Strategy
    print(f"[{datetime.now()}] Triggering Strategy on_bars...")
    
    # We need to ensure the strategy engine is "active" enough to accept orders.
    # PortfolioStrategyEngine doesn't have a "start" flag that blocks orders, 
    # but main_engine must be connected.
    
    # Call on_bars
    try:
        strategy.on_bars(bars_dict)
    except Exception as e:
        print(f"Exception during strategy execution: {e}")
        import traceback
        traceback.print_exc()

    # 6. Wrap up
    print(f"[{datetime.now()}] Execution completed. Waiting for order processing (30s)...")
    time.sleep(30)
    
    # Sync Strategy State?
    # PortfolioStrategy periodically saves. We can force a save if needed or just close.
    # portfolio_engine.close() triggers save.
    print(f"[{datetime.now()}] Closing...")
    main_engine.close()
    print("Done.")

if __name__ == "__main__":
    run_daily_trading()
