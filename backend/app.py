from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime
import requests

app = FastAPI()

# ------------------------------
# CORS Setup
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# GLOBAL VARIABLES
# ------------------------------
df = None
last_modified = None

# ------------------------------
# PATHS
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "NSE_20_stocks_2013_2025_features_target.csv"
)

# HuggingFace Inference API
HF_MODEL_URL = "https://huggingface.co/wanzatess/nse_stock_model"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # optional if private model

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def reload_data_if_needed():
    global df, last_modified
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        return

    current_modified = os.path.getmtime(DATA_PATH)
    if last_modified is None or current_modified > last_modified:
        print(f"📄 Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        last_modified = current_modified
        print(f"✅ Data loaded! {len(df)} records, {df['code'].nunique()} stocks")

def get_latest_stock_data(symbol: str):
    reload_data_if_needed()
    if df is None:
        return None

    stock_df = df[df["code"] == symbol].sort_values("date", ascending=False)
    if stock_df.empty:
        return None

    row = stock_df.iloc[0]
    change_pct = row["changepct"]
    if isinstance(change_pct, str):
        change_pct = float(change_pct.replace("%", ""))

    return {
        "symbol": symbol,
        "name": row["name"],
        "current_price": float(row["day_price"]),
        "previous_price": float(row["previous"]) if pd.notna(row["previous"]) else float(row["day_price"]),
        "change": float(row["change"]) if pd.notna(row["change"]) else 0.0,
        "change_percent": float(change_pct) if pd.notna(change_pct) else 0.0,
        "day_low": float(row["day_low"]) if pd.notna(row["day_low"]) else float(row["day_price"]),
        "day_high": float(row["day_high"]) if pd.notna(row["day_high"]) else float(row["day_price"]),
        "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
        "ma_5": float(row["ma_5"]) if pd.notna(row["ma_5"]) else float(row["day_price"]),
        "ma_10": float(row["ma_10"]) if pd.notna(row["ma_10"]) else float(row["day_price"]),
        "pct_from_12m_low": float(row["pct_from_12m_low"]) if pd.notna(row["pct_from_12m_low"]) else 0.0,
        "pct_from_12m_high": float(row["pct_from_12m_high"]) if pd.notna(row["pct_from_12m_high"]) else 0.0,
        "daily_return": float(row["daily_return"]) if pd.notna(row["daily_return"]) else 0.0,
        "daily_volatility": float(row["daily_volatility"]) if pd.notna(row["daily_volatility"]) else 0.0,
        "last_updated": row["date"].strftime("%Y-%m-%d"),
    }

def predict_with_hf(features: list):
    """Call HuggingFace model API for prediction"""
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    payload = {"inputs": features}

    response = requests.post(HF_MODEL_URL, json=payload, headers=headers, timeout=15)

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"HuggingFace API error: {response.status_code} {response.text}"
        )

    return response.json()

# ------------------------------
# Pydantic Models
# ------------------------------
class PredictRequest(BaseModel):
    symbol: str

# ------------------------------
# API ROUTES
# ------------------------------
@app.get("/")
def root():
    reload_data_if_needed()
    return {
        "status": "healthy" if df is not None else "degraded",
        "data_loaded": df is not None,
        "total_stocks": df["code"].nunique() if df is not None else 0,
    }

@app.get("/stocks")
def get_stocks():
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not available")
    stocks = df.groupby("code")["name"].first().reset_index().to_dict(orient="records")
    return {"stocks": stocks, "count": len(stocks)}

@app.post("/predict")
def predict(request: PredictRequest):
    symbol = request.symbol.upper()
    stock = get_latest_stock_data(symbol)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    # Features in same order as model expects
    features = [
        stock["current_price"],
        stock["ma_5"],
        stock["ma_10"],
        stock["pct_from_12m_low"],
        stock["pct_from_12m_high"],
        stock["daily_return"],
        stock["daily_volatility"]
    ]

    hf_result = predict_with_hf(features)
    return {
        "symbol": symbol,
        "name": stock["name"],
        "prediction": hf_result,
        **stock
    }

# ------------------------------
# STARTUP
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 NSE STOCK PREDICTION API - STARTING UP")
    reload_data_if_needed()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
