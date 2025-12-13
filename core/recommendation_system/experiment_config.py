from datetime import datetime

# Experiment Configurations
# You can define multiple configurations here to test different models or strategies.

CONFIGS = {
    "default_mlp": {
        "description": "Default MLP Model (Trend Following) - Baseline",
        "dataset": {
            "name": "rec_dataset",
            "train_period": ("2022-12-07", "2023-12-31"),
            "valid_period": ("2024-01-01", "2024-06-30"),
            "test_period": ("2024-07-01", "2025-12-12"), 
        },
        "model": {
            "type": "mlp",
            "params": {
                "hidden_sizes": (256,),
                "lr": 0.1,
                "n_epochs": 50,
                "batch_size": 4096,
                "seed": 42
            }
        },
        "strategy": {
            "class_name": "RecStrategy",
            "setting": {
                "buy_count": 2,
                "max_pos": 2,
                "stop_loss": 0.10
            },
            "capital": 1_000_000,
        }
    },
    
    "aggressive_top5": {
        "description": "Aggressive Strategy: Buy Top 5, Higher Risk Tolerance",
        "dataset": {
            "name": "rec_dataset_mlp", # Reuse dataset if periods/features are same
            "train_period": ("2022-12-07", "2023-12-31"),
            "valid_period": ("2024-01-01", "2024-06-30"),
            "test_period": ("2024-07-01", "2025-12-12"),
        },
        "model": {
            "type": "mlp",
            "params": {
                "hidden_sizes": (512, 256), # Deeper model
                "lr": 0.001,
                "n_epochs": 60,
                "batch_size": 4096,
                "seed": 42
            }
        },
        "strategy": {
            "class_name": "RecStrategy",
            "setting": {
                "buy_count": 5,
                "max_pos": 10,
                "stop_loss": 0.15 # Looser stop loss
            },
            "capital": 1_000_000,
        }
    },

    "conservative_hold": {
        "description": "Conservative Strategy: Fewer buys, tighter stop",
        "dataset": {
            "name": "rec_dataset_mlp",
            "train_period": ("2022-12-07", "2023-12-31"),
            "valid_period": ("2024-01-01", "2024-06-30"),
            "test_period": ("2024-07-01", "2025-12-12"),
        },
        "model": {
            "type": "mlp",
            "params": {
                "hidden_sizes": (128,),
                "lr": 0.0005,
                "n_epochs": 50,
                "batch_size": 4096,
                "seed": 42
            }
        },
        "strategy": {
            "class_name": "RecStrategy",
            "setting": {
                "buy_count": 1,
                "max_pos": 5,
                "stop_loss": 0.05
            },
            "capital": 1_000_000,
        }
    }
}
