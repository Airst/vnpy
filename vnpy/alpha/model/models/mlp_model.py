"""
Alpha 预测模型 — 支持 MLP 和 FactorAttentionNetwork 双架构

== 版本演进 ==
初始: MlpNetwork 三层隐藏层, BatchNorm + LeakyReLU + Dropout
V10 Step 3: 新增 FactorAttentionNetwork (FT-Transformer 风格)
V10 Step 3: MlpModel 包装类支持 model_type 切换 ("mlp" / "factor_attention")

== 当前状态 ==
主力架构: FactorAttentionNetwork
- 每个因子投影为 d_token 维 embedding（element-wise weight + bias）
- 可学习 [CLS] token 聚合全局信息
- Pre-Norm Transformer Block: LayerNorm → MultiheadAttention → residual → FFN(GELU)
- 预测头: LayerNorm → Linear → BatchNorm → LeakyReLU → Dropout → Linear → 1
备用架构: MlpNetwork（用于对比实验基线）

== 设计决策 ==
- FT-Transformer 而非 vanilla Transformer: 因子是异构标量，需要 per-feature tokenization
- [CLS] token: 聚合全局截面信息，比平均池化更灵活
- Pre-Norm: 训练更稳定（相比 Post-Norm）
- element-wise embedding（非共享Linear）: 每个因子尺度不同，需独立投影
- early stopping + weight_decay: 防止在有限训练窗口(700天)上过拟合

== 失败记录 ==
- Gate Network 叠加在 Attention 之上: Sharpe 1.76→0.84, 功能冗余+过拟合
- 2层 Attention: 过拟合，1层最优
- d_token=32: 表达能力不足，64最优
"""
import copy
import math
from collections import defaultdict
from typing import Literal, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error      # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim

from vnpy.alpha import (
    AlphaDataset,
    AlphaModel,
    Segment,
    logger
)



class MlpModel(AlphaModel):
    """
    Multi-Layer Perceptron Model

    Alpha factor prediction model implemented using multi-layer perceptron, with main features including:
    1. Building and training multi-layer perceptron neural networks
    2. Predicting Alpha factor values
    3. Model evaluation and feature importance analysis
    4. Support for early stopping and overfitting prevention
    5. Support for MSE loss function
    6. Optional Adam or SGD optimizer
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int] = (256,),
        lr: float = 0.001,
        n_epochs: int = 300,
        batch_size: int = 2000,
        early_stop_rounds: int = 50,
        eval_steps: int = 20,
        optimizer: Literal["sgd", "adam"] = "adam",
        weight_decay: float = 0.0,
        device: str = "cpu",
        seed: int | None = None,
        model_type: str = "mlp",
        d_token: int = 32,
        n_heads: int = 4,
        n_attn_layers: int = 1,
        d_ffn: int = 64,
        head_hidden: int = 64,
        attn_dropout: float = 0.15,
        ffn_dropout: float = 0.15,
        head_dropout: float = 0.10,
    ) -> None:
        """
        Initialize MLP model

        Parameters
        ----------
        input_size : int, default 360
            Input feature dimension
        hidden_sizes : tuple[int], default (256,)
            Number of neurons in hidden layers
        lr : float, default 0.001
            Learning rate
        n_epochs : int, default 300
            Maximum training steps
        batch_size : int, default 2000
            Number of samples per batch
        early_stop_rounds : int, default 50
            Early stopping rounds, training stops if validation loss doesn't improve within these rounds
        eval_steps : int, default 20
            Evaluate model every this many steps
        optimizer : Literal["sgd", "adam"], default "adam"
            Optimizer type, options are "sgd" or "adam"
        weight_decay : float, default 0.0
            L2 regularization coefficient
        seed : Optional[int], optional
            Random seed for reproducibility
        device : str, default "cpu"
            Training device
        """
        # Save model hyperparameters
        self.input_size: int = input_size
        self.hidden_sizes: tuple[int] = hidden_sizes
        self.lr: float = lr
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.early_stop_rounds: int = early_stop_rounds
        self.eval_steps: int = eval_steps
        self.device: str = device
        self.fitted: bool = False
        self.feature_names: list[str] = []
        self.best_step: int | None = None

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Set loss function type
        self._scorer = mean_squared_error

        # Initialize model
        self.model_type = model_type
        if model_type == "factor_attention":
            self.model: nn.Module = FactorAttentionNetwork(
                input_size=input_size,
                d_token=d_token,
                n_heads=n_heads,
                n_layers=n_attn_layers,
                d_ffn=d_ffn,
                head_hidden=head_hidden,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                head_dropout=head_dropout,
            )
        else:
            self.model: nn.Module = MlpNetwork(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
            )

        # Move model to specified device
        self.model = self.model.to(device)

        # Set optimizer
        optimizer_name = optimizer.lower()
        if optimizer_name == "adam":
            self.optimizer: optim.Optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        # Set learning rate scheduler
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0.00001,
            eps=1e-08,
        )

    def fit(
        self,
        dataset: AlphaDataset,
        evaluation_results: dict | None = None,
    ) -> None:
        """
        Train the multi-layer perceptron model

        Trains the MLP model using the given dataset, with main steps including:
        1. Preparing training and validation data
        2. Iteratively training for multiple steps
        3. Evaluating model performance at fixed intervals
        4. Implementing early stopping to prevent overfitting

        Parameters
        ----------
        dataset : AlphaDataset
            Dataset object containing training data
        evaluation_results : dict
            Dictionary for storing evaluation metrics during training
        """
        # Initialize a new dictionary if evaluation_results is None
        if evaluation_results is None:
            evaluation_results = {}

        # Dictionary to store training and validation data
        train_valid_data: dict[str, dict] = defaultdict(dict)

        # Process training and validation sets separately
        for segment in [Segment.TRAIN, Segment.VALID]:
            # Get learning data and sort by time and trading code
            df: pl.DataFrame = dataset.fetch_learn(segment)
            df = df.sort(["datetime", "vt_symbol"])

            # Extract features and labels
            features = df.select(df.columns[2: -1]).to_numpy()
            labels = np.array(df["label"])

            # Store feature and label data
            train_valid_data["x"][segment] = torch.from_numpy(features).float().to(self.device)
            train_valid_data["y"][segment] = torch.from_numpy(labels).float().to(self.device)

            # Initialize evaluation results list
            evaluation_results[segment] = []

        # Get feature names
        df = dataset.fetch_learn(Segment.TRAIN)
        self.feature_names = df.columns[2:-1]

        # Initialize training state
        early_stop_count: int = 0           # Number of steps without performance improvement
        train_loss: float = 0               # Current training loss
        best_valid_score: float = np.inf    # Best validation loss
        best_params = None                  # Best model parameters

        train_samples: int = train_valid_data["y"][Segment.TRAIN].shape[0]

        # Iterate through training steps
        for step in range(1, self.n_epochs + 1):
            # Check if early stopping condition is met
            if early_stop_count >= self.early_stop_rounds:
                logger.info("达到早停条件,训练结束")
                break

            # Train one batch
            batch_loss = self._train_step(train_valid_data, train_samples)
            train_loss += batch_loss

            # Periodically evaluate the model
            if step % self.eval_steps == 0 or step == self.n_epochs:
                early_stop_count, best_valid_score, best_params = self._evaluate_step(
                    train_valid_data,
                    evaluation_results,
                    step,
                    train_loss,
                    early_stop_count,
                    best_valid_score
                )
                train_loss = 0

        # Mark model as trained
        self.fitted = True

        # Load best model parameters
        if best_params:
            self.model.load_state_dict(best_params)

    def _train_step(
        self,
        train_valid_data: dict[str, dict[Segment, torch.Tensor]],
        train_samples: int
    ) -> float:
        """
        Execute one training step

        Parameters
        ----------
        train_valid_data : dict
            Training and validation data
        train_samples : int
            Number of training samples

        Returns
        -------
        float
            Current batch loss value
        """
        batch_loss = AverageMeter()
        self.model.train()
        self.optimizer.zero_grad()

        # Randomly select batch data
        batch_indices = np.random.choice(train_samples, self.batch_size)
        batch_features = train_valid_data["x"][Segment.TRAIN][batch_indices]
        batch_labels = train_valid_data["y"][Segment.TRAIN][batch_indices]

        # Forward and backward propagation
        predictions = self.model(batch_features)
        cur_loss = self._loss_fn(predictions, batch_labels)

        cur_loss.backward()

        # Update model parameters
        self.optimizer.step()
        batch_loss.update(cur_loss.item())

        return batch_loss.val

    def _evaluate_step(
        self,
        train_valid_data: dict[str, dict[Segment, torch.Tensor]],
        evaluation_results: dict[Segment, list[float]],
        step: int,
        train_loss: float,
        early_stop_count: int,
        best_valid_score: float
    ) -> tuple[int, float, dict[str, torch.Tensor] | None]:
        """
        Evaluate current model performance

        Parameters
        ----------
        train_valid_data : dict
            Training and validation data
        evaluation_results : dict
            Evaluation results record
        step : int
            Current training step
        train_loss : float
            Current training loss
        early_stop_count : int
            Count of steps without improvement
        best_valid_score : float
            Best validation loss

        Returns
        -------
        tuple[int, float, dict] | None
            Returns updated early stop count, best validation loss, and best model parameters
        """
        early_stop_count += 1
        train_loss /= self.eval_steps

        # Evaluate model on validation set
        with torch.no_grad():
            self.model.eval()

            data: torch.Tensor = train_valid_data["x"][Segment.VALID]
            pred: torch.Tensor = cast(torch.Tensor, self._predict_batch(data, return_cpu=False))
            valid_loss = self._loss_fn(pred, train_valid_data["y"][Segment.VALID])

            loss_val = valid_loss.item()

        # Record evaluation results
        logger.info(f"[Step {step}]: train_loss {train_loss:.6f}, valid_loss {loss_val:.6f}")
        evaluation_results[Segment.TRAIN].append(train_loss)
        evaluation_results[Segment.VALID].append(loss_val)

        # Update best model if validation performance improves
        best_params = None
        if loss_val < best_valid_score:
            logger.info(f"\t验证集损失从 {best_valid_score:.6f} 降低到 {loss_val:.6f}")
            best_valid_score = loss_val
            self.best_step = step
            early_stop_count = 0
            best_params = copy.deepcopy(self.model.state_dict())

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step(metrics=valid_loss, epoch=step)

        return early_stop_count, best_valid_score, best_params

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss value

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions
        target : torch.Tensor
            Target true values

        Returns
        -------
        torch.Tensor
            Calculated loss value
        """
        pred, target = pred.reshape(-1), target.reshape(-1)
        loss: torch.Tensor = nn.MSELoss()(pred, target)
        return loss

    def _predict_batch(self, data: torch.Tensor, return_cpu: bool = True) -> np.ndarray | torch.Tensor:
        """
        Neural network prediction function

        Parameters
        ----------
        data : torch.Tensor
            Input data
        return_cpu : bool, default True
            Whether to return CPU tensor
        step : Optional[int], optional
            Current training step

        Returns
        -------
        np.ndarray | torch.Tensor
            Model prediction results
        """
        data = data.to(self.device)

        predictions: list[torch.Tensor] = []

        self.model.eval()

        with torch.no_grad():
            batch_size: int = 8096
            for i in range(0, len(data), batch_size):
                x: torch.Tensor = data[i: i + batch_size]
                predictions.append(self.model(x.to(self.device)).detach().reshape(-1))

        if return_cpu:
            return cast(np.ndarray, np.concatenate([pr.cpu().numpy() for pr in predictions]))
        else:
            return torch.cat(predictions, dim=0)

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """
        Model prediction interface

        Parameters
        ----------
        dataset : AlphaDataset
            Prediction dataset
        segment : Segment
            Dataset segment

        Returns
        -------
        np.ndarray
            Prediction result array
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet!")

        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])

        data: np.ndarray = df.select(df.columns[2: -1]).to_numpy()

        return cast(np.ndarray, self._predict_batch(torch.Tensor(data)))

    def _check_tensor_nan(self, tensor: torch.Tensor, name: str) -> None:
        """
        Check if tensor contains NaN values

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to check
        name : str
            Tensor name

        Returns
        -------
        None
        """
        if torch.isnan(tensor).any():
            print(f"NaN values detected: {name}")

    def detail(self) -> pd.DataFrame | None:
        """
        Output MLP model detail information

        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        if not self.fitted:
            logger.info("模型尚未训练，无法显示详细信息")
            return None

        # 显示模型基本信息
        logger.info(f"输入特征维度: {self.input_size}")
        logger.info(f"隐藏层大小: {self.hidden_sizes}")

        # 计算模型总参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型总参数量: {total_params:,}")

        # 显示训练状态信息
        logger.info(f"训练设备: {self.device}")
        logger.info(f"当前学习率: {self.lr}")
        logger.info(f"批次大小: {self.batch_size}")

        # Calculate feature importance
        importance_df = self._calculate_feature_importance()
        return importance_df

    def _calculate_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance

        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        self.model.eval()
        importance_dict = {}

        test_data = torch.randn(1000, self.input_size).to(self.device)
        base_pred = self.model(test_data).detach()

        noise_level = 0.1
        for i, feature_name in enumerate(self.feature_names):
            perturbed_data = test_data.clone()
            perturbed_data[:, i] += torch.randn(1000).to(self.device) * noise_level

            with torch.no_grad():
                new_pred = self.model(perturbed_data)
                importance = torch.std(torch.abs(new_pred - base_pred)).item()
                importance_dict[feature_name] = importance

        df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })
        df = df.sort_values('Importance', ascending=False)
        df = df.set_index('Feature')

        return df


class AverageMeter:
    """
    Class for calculating and storing average and current values

    Attributes
    ----------
    val : float
        Current value
    avg : float
        Average value
    sum : float
        Sum
    count : int
        Count
    """

    def __init__(self) -> None:
        """
        Initialize AverageMeter

        Returns
        -------
        None
        """
        self.reset()

    def reset(self) -> None:
        """
        Reset all statistics

        Returns
        -------
        None
        """
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update statistics

        Parameters
        ----------
        val : float
            Current value
        n : int, default 1
            Current batch size

        Returns
        -------
        None
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FactorAttentionNetwork(nn.Module):
    """
    Factor Self-Attention Network (FT-Transformer style)

    Each factor is independently projected to a d_token-dimensional embedding,
    then Self-Attention models inter-factor interactions. A learnable [CLS] token
    aggregates global interaction information for the final prediction.
    """

    def __init__(
        self,
        input_size: int,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ffn: int = 64,
        head_hidden: int = 64,
        attn_dropout: float = 0.15,
        ffn_dropout: float = 0.15,
        head_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.d_token = d_token

        # === Factor Tokenizer ===
        # Each factor: scalar -> d_token vector via element-wise multiply + bias
        self.token_weight = nn.Parameter(torch.empty(input_size, d_token))
        self.token_bias = nn.Parameter(torch.empty(input_size, d_token))

        # === CLS Token ===
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))

        # === Transformer Blocks ===
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_token),
                'attn': nn.MultiheadAttention(
                    embed_dim=d_token,
                    num_heads=n_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                ),
                'norm2': nn.LayerNorm(d_token),
                'ffn': nn.Sequential(
                    nn.Linear(d_token, d_ffn),
                    nn.GELU(),
                    nn.Dropout(ffn_dropout),
                    nn.Linear(d_ffn, d_token),
                    nn.Dropout(ffn_dropout),
                ),
            }))

        # === Prediction Head ===
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        d = self.d_token
        nn.init.uniform_(self.token_weight, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))
        nn.init.uniform_(self.token_bias, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))
        nn.init.uniform_(self.cls_token, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_size) — scalar factor values
        output: (batch_size, 1)
        """
        batch_size = x.shape[0]

        # === Factor Tokenizer ===
        # x: (B, N) -> (B, N, 1) * (N, d) -> (B, N, d)
        tokens = x.unsqueeze(-1) * self.token_weight.unsqueeze(0) + self.token_bias.unsqueeze(0)

        # === Prepend CLS ===
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, N+1, d)

        # === Transformer Blocks (Pre-Norm) ===
        for block in self.blocks:
            normed = block['norm1'](tokens)
            attn_out, _ = block['attn'](normed, normed, normed)
            tokens = tokens + attn_out

            normed = block['norm2'](tokens)
            ffn_out = block['ffn'](normed)
            tokens = tokens + ffn_out

        # === Extract CLS ===
        cls_output = tokens[:, 0, :]  # (B, d)

        # === Prediction Head ===
        return self.head(cls_output)   # (B, 1)


class MlpNetwork(nn.Module):
    """
    Deep Neural Network Model Structure

    Used to build multi-layer perceptron network structure, supporting multiple hidden layers
    and different activation functions.

    Attributes
    ----------
    network : nn.ModuleList
        List of neural network layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_sizes: tuple[int] = (256,),
        activation: str = "LeakyReLU"
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        input_size : int
            Input feature dimension, i.e., number of features per sample
        output_size : int, default 1
            Output dimension, used for predicting target values
        hidden_sizes : tuple[int], default (256,)
            Tuple of hidden layer neuron counts, e.g., (256, 128) represents two hidden layers
            with 256 and 128 neurons respectively
        activation : str, default "LeakyReLU"
            Activation function type, options:
            - "LeakyReLU": Leaky ReLU function
            - "SiLU": Sigmoid Linear Unit function
        """
        super().__init__()

        # Build network layers
        layers: list[nn.Module] = []
        layer_sizes = [input_size] + list(hidden_sizes)

        # Input layer Dropout
        layers.append(nn.Dropout(0.10))

        # Build hidden layers
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
            # Add a neural network block: linear layer + batch normalization + activation function + dropout
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                self._get_activation(activation),
                nn.Dropout(0.10)
            ])

        # Output layer
        layers.extend([
            nn.Linear(hidden_sizes[-1], output_size)
        ])

        # Combine all layers into a sequence
        self.network = nn.ModuleList(layers)

        # Initialize network weights
        self._initialize_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """
        Get specified activation function layer

        Parameters
        ----------
        name : str
            Activation function name

        Returns
        -------
        nn.Module
            Activation function layer instance

        Raises
        ------
        ValueError
            When an unsupported activation function type is specified
        """
        if name == "LeakyReLU":
            return nn.LeakyReLU(negative_slope=0.1)
        elif name == "SiLU":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function type: {name}")

    def _initialize_weights(self) -> None:
        """
        Initialize network weight parameters

        Uses Kaiming initialization method for all linear layers, which is particularly
        suitable for deep networks using LeakyReLU activation functions.

        Returns
        -------
        None
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.1,                  # LeakyReLU negative slope
                    mode="fan_in",          # Scale using input node count
                    nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation calculation

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor, shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Model output tensor, shape (batch_size, output_size)
        """
        # Pass through all layers in the network sequentially
        for layer in self.network:
            x = layer(x)
        return x
