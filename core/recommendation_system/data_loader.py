import json
from datetime import datetime, timedelta
from functools import partial
import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.alpha.dataset.processor import process_drop_na, process_robust_zscore_norm, process_fill_na

from features import RecommendationDataset

def get_vt_symbols(config_path: str) -> list[str]:
    """Read symbols from config"""
    with open(config_path, "r", encoding="utf-8") as f:
        dl_config = json.load(f)
        
    vt_symbols = []
    for task in dl_config.get("downloads", []):
        s = task["symbol"]
        e = task["exchange"]
        if "." in s:
            code, suffix = s.split(".")
            vt_symbols.append(f"{code}.{e}")
        else:
            vt_symbols.append(f"{s}.{e}")
    return vt_symbols

def get_dataset(
    lab: AlphaLab,
    dataset_name: str,
    vt_symbols: list[str],
    train_period: tuple[str, str],
    valid_period: tuple[str, str],
    test_period: tuple[str, str],
    force_reload: bool = False
) -> RecommendationDataset:
    """
    Get dataset with smart caching:
    1. Try load existing dataset.
    2. Check if coverage is sufficient.
    3. Incrementally update if needed (calculate factors only for new data).
    4. Reprocess (normalization/split).
    """
    
    # 1. Determine required global range
    periods = [train_period, valid_period, test_period]
    dates = []
    for p in periods:
        dates.append(datetime.strptime(p[0], "%Y-%m-%d"))
        dates.append(datetime.strptime(p[1], "%Y-%m-%d"))
    
    req_start_dt = min(dates)
    req_end_dt = max(dates)
    
    print(f"Required Data Range: {req_start_dt.date()} to {req_end_dt.date()}")

    dataset = None
    if not force_reload:
        dataset = lab.load_dataset(dataset_name)
        
    if dataset is not None:
        print(f"Loaded cached dataset: {dataset_name}")
        # Check and update incrementally
        dataset = _update_dataset_incrementally(lab, dataset, vt_symbols, req_start_dt, req_end_dt)
    else:
        print(f"Dataset {dataset_name} not found or force reload. Creating new...")
        dataset = _create_new_dataset(lab, vt_symbols, req_start_dt, req_end_dt)

    # Update Periods (in case config changed)
    dataset.train_period = train_period
    dataset.valid_period = valid_period
    dataset.test_period = test_period
    
    # Save the dataset (with factors) before processing
    # This ensures we save the incremental updates.
    # Note: process_data modifies internal state (learn_df, etc) but usually not self.df (the source).
    # However, saving after process might save the processed views too, which is fine (larger file but faster load).
    
    print("Applying processors and splitting data...")
    _apply_processors_and_process(dataset)
    
    print(f"Saving dataset to cache: {dataset_name}")
    lab.save_dataset(dataset_name, dataset)
    
    return dataset

def _create_new_dataset(
    lab: AlphaLab, 
    vt_symbols: list[str], 
    start_dt: datetime, 
    end_dt: datetime
) -> RecommendationDataset:
    """Load full raw data and calculate factors."""
    EXTENDED_DAYS = 60
    
    print(f"Loading full raw data for {len(vt_symbols)} symbols...")
    df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=start_dt,
        end=end_dt,
        extended_days=EXTENDED_DAYS
    )
    
    if df is None or df.is_empty():
        raise RuntimeError("No data loaded. Please run ingest_data.py first.")
        
    print("Calculating factors (prepare_data)...")
    # Initialize with dummy periods, will be updated later
    dummy_period = (start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    dataset = RecommendationDataset(
        df=df,
        train_period=dummy_period,
        valid_period=dummy_period,
        test_period=dummy_period
    )
    dataset.prepare_data()
    
    return dataset

def _update_dataset_incrementally(
    lab: AlphaLab,
    dataset: RecommendationDataset,
    vt_symbols: list[str],
    req_start_dt: datetime,
    req_end_dt: datetime
) -> RecommendationDataset:
    """Check coverage and append new data with factors if needed."""
    
    if dataset.df.is_empty():
        return _create_new_dataset(lab, vt_symbols, req_start_dt, req_end_dt)
        
    curr_min_dt = dataset.df["datetime"].min()
    curr_max_dt = dataset.df["datetime"].max()
    
    print(f"Cached Data Range: {curr_min_dt.date()} to {curr_max_dt.date()}")
    
    # Check if we need to prepend (start earlier) -> Full reload for simplicity (or complex prepend logic)
    # If required start is significantly earlier than cache start, we probably should reload to be safe with factors.
    # Tolerance: 5 days
    if req_start_dt < curr_min_dt - timedelta(days=5):
        print(f"Cache starts too late ({curr_min_dt} > {req_start_dt}). Reloading full.")
        return _create_new_dataset(lab, vt_symbols, req_start_dt, req_end_dt)
        
    # Check if we need to append (end later)
    # Adjust req_end_dt to the nearest past weekday (if it's a weekend) to avoid unnecessary queries for non-trading days.
    # 5=Saturday, 6=Sunday.
    adjusted_req_end_dt = req_end_dt
    while adjusted_req_end_dt.weekday() >= 5:
        adjusted_req_end_dt -= timedelta(days=1)

    if adjusted_req_end_dt > curr_max_dt:
        print(f"Incremental update needed: Cache ends {curr_max_dt}, Request ends {req_end_dt} (Adjusted: {adjusted_req_end_dt})")
        
        delta_start = curr_max_dt + timedelta(days=1)
        
        # We need lookback for factors. 
        # We can load [delta_start - 60, req_end]
        EXTENDED_DAYS = 60
        lookback_start = delta_start - timedelta(days=EXTENDED_DAYS)
        
        print(f"Loading raw data for update ({lookback_start.date()} to {req_end_dt.date()})...")
        df_delta = lab.load_bar_df(
            vt_symbols=vt_symbols,
            interval=Interval.DAILY,
            start=lookback_start,
            end=req_end_dt,
            extended_days=0 
        )
        
        if df_delta is not None and not df_delta.is_empty():
            # Calculate factors for delta
            print("Calculating factors for new data...")
            dummy_period = (lookback_start.strftime("%Y-%m-%d"), req_end_dt.strftime("%Y-%m-%d"))
            delta_dataset = RecommendationDataset(
                df=df_delta,
                train_period=dummy_period,
                valid_period=dummy_period,
                test_period=dummy_period
            )
            delta_dataset.prepare_data()
            
            # Filter out lookback period (keep only new days)
            # We want rows where datetime > curr_max_dt
            new_rows = delta_dataset.df.filter(pl.col("datetime") > curr_max_dt)
            
            if not new_rows.is_empty():
                print(f"Appending {len(new_rows)} new rows.")
                dataset.df = pl.concat([dataset.df, new_rows])
                dataset.df = dataset.df.unique(subset=["datetime", "vt_symbol"]).sort(["datetime", "vt_symbol"])
            else:
                print("No new rows after factor calculation (maybe holidays).")
        else:
            print("No new raw data found.")
    else:
        print("Cache covers the requested period.")
        
    return dataset

def _apply_processors_and_process(dataset: RecommendationDataset):
    """Re-apply processors and run process_data."""
    # Clear existing processors to avoid duplication if re-loading
    # Accessing protected member _processors if exists, or just relying on overwriting if key matches.
    # AlphaDataset stores processors in self.processors dict.
    # We can just re-add them.
    
    dataset.add_processor("learn", process_drop_na)
    dataset.add_processor("learn", process_robust_zscore_norm)
    
    dataset.add_processor("infer", process_robust_zscore_norm)
    dataset.add_processor("infer", partial(process_fill_na, fill_value=0))
    
    dataset.process_data()