
import sys
from pathlib import Path
import os
import argparse

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def query_signal_remote(table_name: str, limit: int = 5):
    """
    Generates a script to query data on BigQuant AIStudio remotely and executes it.
    """
    
    # --- Remote Script Generation ---
    remote_script_content = f"""
import dai
import pandas as pd

table_name = "{table_name}"
limit = {limit}

print(f"[Remote] Querying BigQuant DataSource: '{{table_name}}'...")

try:
    # Construct SQL query
    sql = f"SELECT * FROM {{table_name}} LIMIT {{limit}}"
    print(f"[Remote] Executing SQL: {{sql}}")
    
    # Execute query
    # dai.query returns a result object, .df() converts to pandas DataFrame
    df = dai.query(sql, full_db_scan=True).df()
    
    if df is None or df.empty:
        print(f"[Remote] Warning: Query returned no data or table '{{table_name}}' does not exist.")
    else:
        print(f"[Remote] Result (First {{limit}} rows):")
        print(df)
        print(f"[Remote] Total columns: {{df.columns.tolist()}}")
        print(f"[Remote] Data types:\\n{{df.dtypes}}")

except Exception as e:
    print(f"[Remote] Query failed: {{e}}")
    import traceback
    traceback.print_exc()
"""

    # Write to temp file
    temp_dir = Path("build")
    temp_dir.mkdir(exist_ok=True)
    temp_script_path = temp_dir / "bq_query_temp.py"
    
    print(f"[Query] Generating remote script at: {temp_script_path}")
    with open(temp_script_path, "w", encoding="utf-8") as f:
        f.write(remote_script_content)

    # --- Remote Execution ---
    print(f"[Query] Initiating BigQuant AIStudio session...")
    try:
        import bigquant
        
        # Resource Spec ID provided by user or default
        resource_spec_id = "f35dcb36-8155-42cb-8255-ecee63b5a723" 
        print(f"[Query] Starting Studio with spec ID: {resource_spec_id}")
        
        studio = bigquant.aistudio.start(resource_spec_id=resource_spec_id)
        print(f"[Query] ✓ AIStudio started.")

        print(f"[Query] Sending script for execution...")
        output = studio.run(str(temp_script_path.absolute()), is_code=False)
        
        print("\n" + "="*40)
        print("REMOTE EXECUTION OUTPUT")
        print("="*40)
        print(output)
        print("="*40 + "\n")

    except ImportError:
        print("[Query] Error: 'bigquant' SDK not installed locally.")
        print("Please install it using: pip install bigquant")
    except Exception as e:
        print(f"[Query] Remote execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query BigQuant DataSource (Remote Execution)")
    parser.add_argument("-t", "--table", type=str, help="Table name in BigQuant", default="ashare_mlp_signal_v7")
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to display")
    
    args = parser.parse_args()
    
    query_signal_remote(args.table, args.limit)