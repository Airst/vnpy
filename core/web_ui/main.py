import sys
from pathlib import Path
from datetime import datetime, date
import traceback

# Add project root to path to import core modules
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))
# Add recommendation_system to path to fix sibling imports (e.g. 'from ingest_data import ...' in data_manager)
sys.path.append(str(root_path / "core" / "recommendation_system"))

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import polars as pl
import numpy as np
import json

from core.recommendation_system.experiment_config import CONFIGS
from core.recommendation_system.data_manager import DataManager
from core.recommendation_system.trainer import train_model
from core.recommendation_system.backtester import run_backtesting
from core.recommendation_system.predictor import predict_daily
from core.recommendation_system.data_loader import get_vt_symbols

app = FastAPI(title="VNPY Recommendation System")
templates = Jinja2Templates(directory="core/web_ui/templates")

# Helper to serialize Polars/Datetime objects
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, pl.DataFrame):
            return obj.to_dicts()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "configs": list(CONFIGS.keys())
    })

@app.post("/api/update_data")
async def update_data():
    try:
        dm = DataManager()
        dm.check_and_update_all()
        return {"status": "success", "message": "Data updated successfully"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/api/run_backtest")
async def run_backtest_api(request: Request):
    data = await request.json()
    config_name = data.get("config", "default_mlp")
    force_reload = data.get("force", False)
    
    try:
        # 1. Train
        model, dataset = train_model(config_name, force_reload=force_reload)
        
        # 2. Backtest
        engine = run_backtesting(config_name, model, dataset)
        
        # 3. Process Results
        if engine.daily_df is None:
            return {"status": "error", "message": "No daily results generated."}
            
        daily_data = engine.daily_df.to_dicts()
        
        stats = engine.calculate_statistics()
        
        return JSONResponse(content={
            "status": "success",
            "daily_data": json.loads(json.dumps(daily_data, cls=CustomEncoder)),
            "stats": json.loads(json.dumps(stats, cls=CustomEncoder)),
            "trades": [
                {
                    "date": t.datetime.isoformat(),
                    "symbol": t.symbol,
                    "direction": t.direction.value,
                    "offset": t.offset.value,
                    "price": t.price,
                    "volume": t.volume
                } 
                for t in engine.get_all_trades()
            ]
        })
        
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/api/run_prediction")
async def run_prediction_api(request: Request):
    data = await request.json()
    config_name = data.get("config", "default_mlp")
    
    try:
        final_df = predict_daily(config_name)
        
        if final_df.is_empty():
            return {"status": "warning", "message": "No predictions found."}
            
        # Format for display
        # Sort by signal
        final_df = final_df.sort("signal", descending=True).head(5)
        
        results = final_df.to_dicts()
        
        return JSONResponse(content={
            "status": "success",
            "data": json.loads(json.dumps(results, cls=CustomEncoder))
        })
        
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
