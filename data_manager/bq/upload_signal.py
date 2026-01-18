import sys
from pathlib import Path
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import argparse
from vnpy.alpha.lab import AlphaLab
import polars as pl
import pandas as pd

def upload_signal(signal_name: str, table_name: str, lab_path: str):
    # Initialize AlphaLab
    lab = AlphaLab(lab_path)
    
    print(f"[Upload] Loading signal '{signal_name}' from '{lab_path}'...")
    df = lab.load_signal(signal_name)
    
    if df is None:
        print(f"[Upload] Error: Signal '{signal_name}' not found.")
        # List available signals
        signals = lab.list_all_signals()
        print(f"[Upload] Available signals: {signals}")
        return

    print(f"[Upload] Loaded {len(df)} rows. Converting to pandas...")
    # Convert to pandas
    pdf = df.to_pandas()
    
    # Rename columns to match BigQuant convention
    # vnpy: datetime, vt_symbol -> BigQuant: date, instrument
    rename_map = {}
    if "datetime" in pdf.columns:
        rename_map["datetime"] = "date"
    if "vt_symbol" in pdf.columns:
        rename_map["vt_symbol"] = "instrument"
    
    if rename_map:
        print(f"[Upload] Renaming columns: {rename_map}")
        pdf.rename(columns=rename_map, inplace=True)
        
    # Ensure date is datetime64[ns]
    if "date" in pdf.columns:
        pdf["date"] = pd.to_datetime(pdf["date"])
    
    print(f"[Upload] Uploading to BigQuant DataSource: '{table_name}'...")
    
    try:
        from bigquant import dai
    except ImportError:
        print("[Upload] Error: 'bigquant' SDK not installed.")
        print("Please install it using: pip install bigquant")
        return

    try:
        # Upload using write_bdb (Class Method) with optimizations
        ds = dai.DataSource.write_bdb(
            data=pdf,
            id=table_name,
            partitioning=["date"],        # Partition by date for query performance
            indexes=["instrument"],       # Index instrument for lookups
            unique_together=["date", "instrument"], # Ensure uniqueness
            on_duplicates="last",         # Keep latest if duplicates exist
            sort_by=[("date", "ascending"), ("instrument", "ascending")],
            overwrite=True                # Overwrite existing datasource if it exists
        )
             
        print(f"[Upload] Successfully uploaded to table '{table_name}'.")
        print(f"  DataSource ID: {ds.id}")
        print(f"  Rows: {len(pdf)}")
        
    except Exception as e:
        print(f"[Upload] Upload failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload VNPY Alpha Signal to BigQuant")
    parser.add_argument("-s", "--signal_name", type=str, help="Name of the signal file in AlphaLab (without extension)", default="ashare_mlp_signal_v6")
    parser.add_argument("--table", type=str, help="Target table name in BigQuant (default: same as signal_name)", default=None)
    parser.add_argument("--lab_path", type=str, default="core/alpha_db", help="Path to AlphaLab directory")
    
    args = parser.parse_args()

    if not args.signal_name:
        args.signal_name = "ashare_mlp_signal_v6"
    
    target_table = args.table if args.table else args.signal_name
    
    upload_signal(args.signal_name, target_table, args.lab_path)
