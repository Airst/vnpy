import argparse
from vnpy.trader.ui import create_qapp

# Import local modules
from ui import BacktestResultDialog
from trainer import train_model
from backtester import run_backtesting
from data_manager import DataManager


def run_backtest(config_name: str = "default_mlp", no_ui: bool = False, force_reload: bool = False):
    # 0. Check & Update Data
    dm = DataManager()
    dm.check_and_update_all(force=force_reload)

    # 1. Train Model
    try:
        model, dataset = train_model(config_name, force_reload=force_reload)
    except Exception as e:
        print(f"Training failed: {e}")
        return

    # 2. Run Backtest
    try:
        engine = run_backtesting(config_name, model, dataset)
    except Exception as e:
        print(f"Backtesting failed: {e}")
        return
    
    # 3. Visualization
    if not no_ui:
        print("Displaying Results in GUI...")
        
        if engine.daily_df is not None:
            qapp = create_qapp()
            dialog = BacktestResultDialog(engine.daily_df, engine.get_all_trades())
            dialog.show()
            qapp.exec()
        else:
            print("No daily result to show.")
    else:
        print("Backtest finished. GUI skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stock Recommendation Backtest")
    parser.add_argument("-c", "--config", type=str, default="default_mlp", help="Configuration name to run (from experiment_config.py)")
    parser.add_argument("--no-ui", action="store_true", help="Run without GUI (headless mode)")
    parser.add_argument("--force", action="store_true", help="Force reload raw data and dataset")
    args = parser.parse_args()
    
    run_backtest(args.config, args.no_ui, args.force)
