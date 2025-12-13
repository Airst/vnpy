import polars as pl
import torch
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.model.models.mlp_model import MlpModel
from vnpy.alpha.dataset import AlphaDataset

from experiment_config import CONFIGS
from data_loader import get_vt_symbols, load_raw_data, create_dataset

def train_model(config_name: str = "default_mlp") -> tuple[MlpModel, AlphaDataset]:
    if config_name not in CONFIGS:
        raise ValueError(f"Config '{config_name}' not found available configs: {list(CONFIGS.keys())}")

    cfg = CONFIGS[config_name]
    print(f"Running Experiment: {config_name}")
    print(f"Description: {cfg['description']}")
    
    # 1. Configuration
    lab_path = "core/alpha_db"
    config_path = "core/data_download/download_config.json"
    
    dataset_cfg = cfg["dataset"]
    train_period = dataset_cfg["train_period"]
    valid_period = dataset_cfg["valid_period"]
    test_period  = dataset_cfg["test_period"]

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    lab = AlphaLab(lab_path)
    
    # 2. Load Data
    vt_symbols = get_vt_symbols(config_path)

    dataset_name = dataset_cfg.get("name", "rec_dataset")
    dataset = lab.load_dataset(dataset_name)

    if dataset:
        print(f"Loaded cached dataset: {dataset_name}")
        print("To force refresh, delete the cache file in core/alpha_db/dataset/")
    else:
        df = load_raw_data(lab, vt_symbols, train_period[0], test_period[1])
        
        if df is None or df.is_empty():
            raise RuntimeError("No data loaded. Please run ingest_data.py first.")

        # 3. Prepare Dataset
        dataset = create_dataset(df, train_period, valid_period, test_period)
        
        print(f"Saving dataset to cache: {dataset_name}")
        lab.save_dataset(dataset_name, dataset)
    
    # 4. Train Model
    model_cfg = cfg["model"]
    print(f"Training {model_cfg['type']} Model on {device}...")
    
    # learn_df columns: datetime, vt_symbol, feature1, feature2, ..., label
    feature_count = len(dataset.learn_df.columns) - 3 # datetime, vt_symbol, label
    print(f"Feature count: {feature_count}")

    model_params = model_cfg["params"]
    
    # Initialize Model (Assuming MLP for now, expand if needed)
    if model_cfg["type"] == "mlp":
        model = MlpModel(
            input_size=feature_count,
            hidden_sizes=model_params["hidden_sizes"],
            lr=model_params["lr"],
            n_epochs=model_params["n_epochs"],
            batch_size=model_params["batch_size"],
            device=device,
            seed=model_params["seed"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")
    
    model.fit(dataset)
    model.detail()
    
    # Save Model
    model_name = f"{config_name}_model"
    print(f"Saving model to {model_name}...")
    lab.save_model(model_name, model)
    
    return model, dataset
