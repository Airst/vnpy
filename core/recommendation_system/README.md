# A-Share Stock Recommendation System

This system uses `vnpy.alpha` to build a stock recommendation model based on historical data.

## Structure

- `ingest_data.py`: Imports data from the `vnpy` database (downloaded via `core/data_download`) into the `AlphaLab` parquet storage.
- `features.py`: Defines the alpha factors (Momentum, Volatility, RSI) used for recommendation.
- `strategy.py`: Defines the backtesting strategy that buys the top-ranked stocks based on the model's prediction.
- `run_backtest.py`: The main script to train the model, generate signals, run the backtest, and display performance charts.

## Usage

1.  **Prepare Data:**
    Ensure you have downloaded data using `core/data_download/download_data.py`.
    Then, run the ingestion script to format it for the Alpha system:
    ```bash
    python core/recommendation_system/ingest_data.py
    ```

2.  **Run Backtest:**
    Train the model and run the strategy backtest:
    ```bash
    python core/recommendation_system/run_backtest.py
    ```
    This will output performance statistics and open a GUI window with the equity curve and drawdown charts.

## Configuration

- **Data Config:** `core/download_config.json`
- **Strategy Parameters:** Edit `RecStrategy` in `strategy.py` or the settings in `run_backtest.py` (e.g., `top_k`, `capital`).
- **Model:** Uses a LASSO regression model. You can modify `features.py` to add more technical indicators.
