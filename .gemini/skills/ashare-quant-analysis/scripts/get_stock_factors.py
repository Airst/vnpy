import sys
import os
import argparse
from pathlib import Path
import polars as pl
import pandas as pd
import traceback

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Calculators dynamically to avoid import errors if some are missing
from core.alpha.v3_factor_calculator import V3FactorCalculator
from core.alpha.v4_factor_calculator import V4FactorCalculator
from core.alpha.v5_factor_calculator import V5FactorCalculator
from core.alpha.v6_factor_calculator import V6FactorCalculator
from core.alpha.v7_factor_calculator import V7FactorCalculator
from core.alpha.v8_factor_calculator import V8FactorCalculator
from core.alpha.data_loader import DataLoader
from vnpy.alpha.lab import AlphaLab

VERSION_CONFIG = {
    "v3": V3FactorCalculator,
    "v4": V4FactorCalculator,
    "v5": V5FactorCalculator,
    "v6": V6FactorCalculator,
    "v7": V7FactorCalculator,
    "v8": V8FactorCalculator,
}

def main():
    parser = argparse.ArgumentParser(description="Get Stock Factors")
    parser.add_argument("--vt_symbol", required=True, help="VT Symbol (e.g., 000001.SZ)")
    parser.add_argument("--version", required=True, help="Alpha Version (e.g., v8)")
    parser.add_argument("--start_date", default="2023-01-01", help="Start Date")
    parser.add_argument("--end_date", default="2025-01-01", help="End Date")
    
    args = parser.parse_args()
    
    version = args.version.lower()
    if not version.startswith("v"):
        version = "v" + version
        
    if version not in VERSION_CONFIG:
        print(f"Error: Unknown version {version}")
        sys.exit(1)
        
    calc_class = VERSION_CONFIG[version]
    calculator = calc_class()
    
    print(f"Initializing DataLoader for {args.vt_symbol}...")
    lab = AlphaLab(str(PROJECT_ROOT / "core/alpha_db"))
    loader = DataLoader(lab)
    
    print(f"Loading data from {args.start_date} to {args.end_date}...")
    # DataLoader expects list of symbols
    df = loader.load_ashare_data([args.vt_symbol], args.start_date, args.end_date)
    
    if df.is_empty():
        print("No data found.")
        return

    print(f"Calculating factors using {version}...")
    try:
        factor_df = calculator.calculate_features(df)
        
        # Output result
        print("-" * 30)
        print(f"Factor Data for {args.vt_symbol} ({version})")
        print(f"Shape: {factor_df.shape}")
        
        # Show last 5 rows
        print(factor_df.tail(5))
        
        # Save to csv for analysis
        output_file = f"factors_{args.vt_symbol}_{version}.csv"
        factor_df.write_csv(output_file)
        print(f"Saved to {output_file}")
        
    except Exception as e:
        print(f"Error calculating factors: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
