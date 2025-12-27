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

from core.engine import RecommendationEngine

app = FastAPI()
engine = RecommendationEngine()

# API Routes
class BacktestRequest(BaseModel):
    strategy_name: str
    start_date: str
    end_date: str
    setting: dict = {}

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
            setting=req.setting
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def run_prediction(req: PredictionRequest):
    try:
        predictions = engine.run_prediction(
            strategy_name=req.strategy_name,
            setting=req.setting
        )
        return {"results": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting backtest manually...")
    run_backtest(BacktestRequest(
        strategy_name="MultiFactorStrategy",
        start_date="20221209",
        end_date="20251219",
        setting={
            "max_holdings": 5
        }
    ))
    # trigger_ingest()
