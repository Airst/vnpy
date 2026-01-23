import sys
from pathlib import Path
import os
import tempfile
import json

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import argparse
from vnpy.alpha.lab import AlphaLab
import pandas as pd

def upload_signal(signal_name: str, table_name: str, lab_path: str):
    # Initialize AlphaLab
    lab = AlphaLab(lab_path)
    
    print(f"[Upload] Loading signal '{signal_name}' from '{lab_path}'...")
    df = lab.load_signal(signal_name)
    
    if df is None:
        print(f"[Upload] Error: Signal '{signal_name}' not found.")
        # List available signals
        try:
            signals = lab.list_all_signals()
            print(f"[Upload] Available signals: {signals}")
        except:
            pass
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

    # --- Filtering Logic ---
    if "date" in pdf.columns:
        latest_date = pdf["date"].max()
        print(f"[Upload] Latest date found: {latest_date}")
        daily_df = pdf[pdf["date"] == latest_date].copy()
    else:
        print("[Upload] Warning: 'date' column not found, using all data.")
        daily_df = pdf.copy()

    # Sort by score and take top 5
    # Prefer 'final_signal', then 'total_score', then just take first 5
    score_col = None
    if "final_signal" in daily_df.columns:
        score_col = "final_signal"
    elif "total_score" in daily_df.columns:
        score_col = "total_score"
    
    if score_col:
        print(f"[Upload] Sorting by '{score_col}' descending to get Top 5...")
        daily_df = daily_df.sort_values(by=score_col, ascending=False).head(5)
    else:
        print("[Upload] Warning: No score column ('final_signal' or 'total_score') found. Taking first 5 rows.")
        daily_df = daily_df.head(5)
    
    print(f"[Upload] Data to upload (Top 5):\n{daily_df}")

    # Prepare data for embedding
    # Convert timestamps to string for JSON serialization compatibility in the generated script
    if "date" in daily_df.columns:
        daily_df["date"] = daily_df["date"].astype(str)
    
    data_records = daily_df.to_dict(orient="records")

    # --- Remote Script Generation ---
    # We embed the data directly into the script
    remote_script_content = f"""
import pandas as pd
import dai

print("[Remote] Script started.")

# Embedded Data
data = {json.dumps(data_records, default=str)}

print(f"[Remote] Received {{len(data)}} rows.")

if not data:
    print("[Remote] No data to upload.")
else:
    df = pd.DataFrame(data)
    # Convert date back to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    print(f"[Remote] DataFrame Info:")
    print(df.info())
    print(df.head())

    table_name = "{table_name}"
    print(f"[Remote] Uploading to BigQuant DataSource: '{{table_name}}'...")

    try:
        # Upload using write_bdb
        # Using overwrite=False to APPEND daily top 5.
        # Ensure 'date' partition is used.
        ds = dai.DataSource.write_bdb(
            data=df,
            id=table_name,
            partitioning=["date"],
            unique_together=["date", "instrument"],
            on_duplicates="last",
            sort_by=[("date", "ascending"), ("instrument", "ascending")],
            overwrite=False 
        )
             
        print(f"[Remote] Successfully uploaded to table '{{table_name}}'.")
        print(f"[Remote] DataSource ID: {{ds.id}}")
        
    except Exception as e:
        print(f"[Remote] Upload failed: {{e}}")
        import traceback
        traceback.print_exc()

print("[Remote] Script finished.")
"""

    # Write to temp file
    temp_dir = Path("build")
    temp_dir.mkdir(exist_ok=True)
    temp_script_path = temp_dir / "bq_upload_temp.py"
    
    print(f"[Upload] Generating remote script at: {temp_script_path}")
    with open(temp_script_path, "w", encoding="utf-8") as f:
        f.write(remote_script_content)

    # --- Remote Execution ---
    print(f"[Upload] Initiating BigQuant AIStudio session...")
    try:
        import bigquant
        
        # Resource Spec ID provided by user
        resource_spec_id = "f35dcb36-8155-42cb-8255-ecee63b5a723" 
        print(f"[Upload] Starting Studio with spec ID: {resource_spec_id}")
        
        studio = bigquant.aistudio.start(resource_spec_id=resource_spec_id)
        print(f"[Upload] ✓ AIStudio started.")

        print(f"[Upload] Sending script for execution...")
        output = studio.run(str(temp_script_path.absolute()), is_code=False)
        
        print("\n" + "="*40)
        print("REMOTE EXECUTION OUTPUT")
        print("="*40)
        print(output)
        print("="*40 + "\n")

    except ImportError:
        print("[Upload] Error: 'bigquant' SDK not installed locally.")
        print("Please install it using: pip install bigquant")
    except Exception as e:
        print(f"[Upload] Remote execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload VNPY Alpha Signal to BigQuant (Remote Execution)")
    parser.add_argument("-s", "--signal_name", type=str, help="Name of the signal file in AlphaLab (without extension)", default="ashare_mlp_signal_v7")
    parser.add_argument("--table", type=str, help="Target table name in BigQuant (default: same as signal_name)", default=None)
    parser.add_argument("--lab_path", type=str, default="core/alpha_db", help="Path to AlphaLab directory")
    
    args = parser.parse_args()

    if not args.signal_name:
        args.signal_name = "ashare_mlp_signal_v7"
    
    target_table = args.table if args.table else args.signal_name
    
    upload_signal(args.signal_name, target_table, args.lab_path)
