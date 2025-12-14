import argparse
from vnpy.trader.ui import create_qapp

from ui import RecommendationDialog
import argparse
from vnpy.trader.ui import create_qapp

from ui import RecommendationDialog
from predictor import predict_daily
from data_manager import DataManager


def run_prediction(config_name: str = "default_mlp", force_reload: bool = False):
    # 0. Check & Update Data
    dm = DataManager()
    dm.check_and_update_all(force=force_reload)

    # 1. Predict
    try:
        final_df = predict_daily(config_name, force_reload=force_reload)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    if final_df.is_empty():
        print("No predictions generated.")
        return

    # 2. Show Results
    print("Displaying...")
    qapp = create_qapp()
    dialog = RecommendationDialog(final_df, title=f"Daily Recommendations ({config_name})")
    dialog.show()
    qapp.exec()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="default_mlp")
    parser.add_argument("--force", action="store_true", help="Force reload raw data from DB")
    args = parser.parse_args()
    
    run_prediction(args.config, force_reload=args.force)