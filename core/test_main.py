import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import sys
import os
from pathlib import Path

# Ensure project root is in path
# core/main.py -> core -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.core_service import CoreService

app = FastAPI()
engine = CoreService()

# API Routes
class BacktestRequest(BaseModel):
    strategy_name: str
    start_date: str
    end_date: str
    setting: dict = {}
    vt_symbols: list = []

class PredictionRequest(BaseModel):
    strategy_name: str
    setting: dict = {}

def get_strategies():
    return {"strategies": engine.get_strategies()}

def get_data_range():
    start, end = engine.get_data_range()
    if start and end:
        return {
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d")
        }
    return {"start": "", "end": ""}

def run_backtest(req: BacktestRequest):
    try:
        start = datetime.strptime(req.start_date, "%Y%m%d")
        end = datetime.strptime(req.end_date, "%Y%m%d")
        
        result = engine.run_backtest(
            strategy_name=req.strategy_name,
            start=start,
            end=end,
            setting=req.setting,
            vt_symbols=req.vt_symbols if req.vt_symbols else None # type: ignore
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting backtest manually...")
    run_backtest(BacktestRequest(
        strategy_name="MultiFactorStrategy",
        start_date="20221209",
        end_date="20251219",
        setting={
            "max_holdings": 5,
            "buy_threshold": 1,
            "sell_threshold": 0.5,
            "signal_name": "ashare_mlp_signal_v3",
        },
        # vt_symbols=["300815.SZSE"]
    ))
    # trigger_ingest()
