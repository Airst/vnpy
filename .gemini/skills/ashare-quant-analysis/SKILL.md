---
name: ashare-quant-analysis
description: Analyzes A-share stocks for quantitative trading factor optimization. Use when the user wants to analyze a specific stock's factors, model signals, or improve the win rate of a specific alpha version (e.g., "Analyze 000001 with v8").
---

# A-Share Quant Analysis

## 1. Context Identification

Identify the target stock and alpha version from the user's request.
- **Stock**: Look for stock codes (e.g., `000001`, `600519`). Append `.SZ` for codes starting with 0 or 3, and `.SH` for codes starting with 6 if the user provides only the number. The system expects `vt_symbol` format (e.g., `000001.SZ`).
- **Version**: Look for version identifiers like `v8`, `v7`. Default to `v8` if not specified.

## 2. Data Gathering

Execute the following scripts to gather necessary data. 
**CRITICAL**: You MUST use the specific python interpreter: `/home/airst/Workspace/.venv/bin/python`.

### 2.1 Get Market Data
Get basic OHLC and turnover data.
```bash
/home/airst/Workspace/.venv/bin/python .skills/ashare-quant-analysis/scripts/get_market_data.py --vt_symbol <VT_SYMBOL>
```

### 2.2 Get Factor Data
Calculate alpha factors for the stock using the specified version.
```bash
/home/airst/Workspace/.venv/bin/python .skills/ashare-quant-analysis/scripts/get_stock_factors.py --vt_symbol <VT_SYMBOL> --version <VERSION>
```

### 2.3 Get Model Signals
Retrieve the MLP model's output scores.
```bash
/home/airst/Workspace/.venv/bin/python .skills/ashare-quant-analysis/scripts/get_model_signals.py --vt_symbol <VT_SYMBOL> --version <VERSION>
```

## 3. Code Analysis

Read the factor calculator code to understand how factors are computed.
- Read `core/alpha/<version>_factor_calculator.py`.

## 4. Analysis & Optimization

1.  **Correlate**: Compare the generated factors and model scores with the actual market price movements (from step 2.1).
2.  **Identify Weakness**: Look for divergences (e.g., high score but price dropped, or low score but price rose).
3.  **Propose Improvements**: Suggest changes to the factor calculator (e.g., new factors, adjusting weights, changing logic) to better capture the stock's behavior.
4.  **Implement**: If the user agrees, use the `replace` tool to modify the factor calculator code.

## 5. Verification
After modification, run the `get_stock_factors.py` script again to verify the new factors are calculated correctly and values have changed as expected.