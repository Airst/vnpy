from typing import cast

import numpy as np
import polars as pl
from pytorch_tabnet.tab_model import TabNetRegressor

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model import AlphaModel


class TabNetModel(AlphaModel):
    """TabNet regression model for alpha prediction"""

    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-4,
        learning_rate: float = 0.02,
        batch_size: int = 4096,
        max_epochs: int = 200,
        patience: int = 30,
        seed: int = 42,
        device: str = "auto",
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.device = device

        self.model: TabNetRegressor | None = None

    def fit(self, dataset: AlphaDataset) -> None:
        import torch
        if self.device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.device

        train_df: pl.DataFrame = dataset.fetch_learn(Segment.TRAIN)
        train_df = train_df.sort(["datetime", "vt_symbol"])
        X_train = train_df.select(train_df.columns[2:-1]).to_numpy().astype(np.float32)
        y_train = np.array(train_df["label"], dtype=np.float32).reshape(-1, 1)

        valid_df: pl.DataFrame = dataset.fetch_learn(Segment.VALID)
        valid_df = valid_df.sort(["datetime", "vt_symbol"])
        X_valid = valid_df.select(valid_df.columns[2:-1]).to_numpy().astype(np.float32)
        y_valid = np.array(valid_df["label"], dtype=np.float32).reshape(-1, 1)

        # Replace NaN with 0 for TabNet
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_valid = np.nan_to_num(X_valid, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_valid = np.nan_to_num(y_valid, nan=0.0)

        self.model = TabNetRegressor(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            optimizer_params={"lr": self.learning_rate},
            seed=self.seed,
            device_name=device_name,
            verbose=10,
        )

        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=["valid"],
            eval_metric=["mse"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
        )

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])
        data = df.select(df.columns[2:-1]).to_numpy().astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)

        result = self.model.predict(data)
        return result.flatten()
