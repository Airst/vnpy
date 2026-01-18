import sys
from pathlib import Path
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import argparse
import pandas as pd

def query_signal(table_name: str, limit: int = 5):
    print(f"[Query] Querying BigQuant DataSource: '{table_name}'...")
    
    try:
        from bigquant import dai
    except ImportError:
        print("[Query] Error: 'bigquant' SDK not installed.")
        print("Please install it using: pip install bigquant")
        return

    try:
        # Construct SQL query
        sql = f"SELECT * FROM {table_name} LIMIT {limit}"
        print(f"[Query] Executing SQL: {sql}")
        
        # Execute query
        # dai.query returns a result object, .df() converts to pandas DataFrame
        df = dai.query(sql).df()
        
        if df is None or df.empty:
            print(f"[Query] Warning: Query returned no data or table '{table_name}' does not exist.")
        else:
            print(f"\n[Query] Result (First {limit} rows):")
            print(df)
            print(f"\n[Query] Total columns: {df.columns.tolist()}")
            print(f"[Query] Data types:\n{df.dtypes}")

    except Exception as e:
        print(f"[Query] Query failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query BigQuant DataSource for verification")
    parser.add_argument("-t", "--table", type=str, help="Table name in BigQuant", default="ashare_mlp_signal_v6")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to display")
    
    args = parser.parse_args()
    
    query_signal(args.table, args.limit)
