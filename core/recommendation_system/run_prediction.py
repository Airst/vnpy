import argparse
from vnpy.trader.ui import create_qapp

from ui import RecommendationDialog
from predictor import predict_daily


def run_prediction(config_name: str = "default_mlp"):
    # 1. Predict
    try:
        final_df = predict_daily(config_name)
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
    args = parser.parse_args()
    
    run_prediction(args.config)