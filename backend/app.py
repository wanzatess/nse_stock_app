from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime
import requests
from pathlib import Path

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
model = None
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

# Render filesystem path (DO NOT CHANGE)
MODEL_DIR = "/opt/render/project/src/backend/models"
MODEL_FILENAME = "stock_model_compressed.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

MODEL_URL = (
    "https://huggingface.co/wanzatess/nse_stock_model/"
    "resolve/main/stock_model_compressed.pkl"
)

# ------------------------------
# HELPERS
# ------------------------------
def ensure_model_directory():
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✅ Model directory ready: {MODEL_DIR}")

def download_model():
    ensure_model_directory()

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model already exists ({size_mb:.2f} MB)")
        return

    print("📥 Downloading model from Hugging Face...")

    try:
        with requests.get(MODEL_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model downloaded successfully ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"❌ Model download failed: {e}")

def load_model():
    global model

    if model is not None:
        return True

    if not os.path.exists(MODEL_PATH):
        print("❌ Model file missing after download")
        return False

    try:
        print(f"📦 Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

        if not hasattr(model, "predict"):
            raise Exception("Loaded object is not a valid ML model")

        print("✅ Model loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Model load failed: {e}")
        model = None
        return False

def reload_data_if_needed():
    global df, last_modified

    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        return

    modified = os.path.getmtime(DATA_PATH)

    if last_modified is None or modified > last_modified:
        print(f"📄 Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        last_modified = modified
        print(f"✅ Data loaded: {len(df)} rows, {df['code'].nunique()} stocks")

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
        "ma_5": float(row["ma_5"]),
        "ma_10": float(row["ma_10"]),
        "pct_from_12m_low": float(row["pct_from_12m_low"]),
        "pct_from_12m_high": float(row["pct_from_12m_high"]),
        "daily_return": float(row["daily_return"]),
        "daily_volatility": float(row["daily_volatility"]),
        "last_updated": row["date"].strftime("%Y-%m-%d"),
    }

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
        "status": "healthy" if model and df is not None else "degraded",
        "model_loaded": model is not None,
        "data_loaded": df is not None,
        "total_stocks": df["code"].nunique() if df is not None else 0,
    }

@app.get("/stocks")
def get_stocks():
    reload_data_if_needed()

    if df is None:
        raise HTTPException(status_code=503, detail="Data not available")

    stocks = (
        df.groupby("code")["name"]
        .first()
        .reset_index()
        .to_dict(orient="records")
    )

    return {"stocks": stocks, "count": len(stocks)}

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    stock = get_latest_stock_data(request.symbol.upper())

    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    features = [[
        stock["current_price"],
        stock["ma_5"],
        stock["ma_10"],
        stock["pct_from_12m_low"],
        stock["pct_from_12m_high"],
        stock["daily_return"],
        stock["daily_volatility"],
    ]]

    prediction = model.predict(features)[0]

    return {
        "symbol": stock["symbol"],
        "name": stock["name"],
        "prediction": prediction,
        **stock,
    }

# ------------------------------
# FASTAPI STARTUP (RENDER SAFE)
# ------------------------------
@app.on_event("startup")
def startup_event():
    print("\n" + "=" * 70)
    print("🚀 NSE STOCK PREDICTION API - STARTUP")
    print("=" * 70)

    reload_data_if_needed()
    download_model()
    model_loaded = load_model()

    print("=" * 70)
    print(f"Status: {'✅ READY' if model_loaded and df is not None else '⚠️ DEGRADED'}")
    print(f"Data: {'✅ Loaded' if df is not None else '❌ Missing'}")
    print(f"Model: {'✅ Loaded' if model_loaded else '❌ Missing'}")
    print("=" * 70 + "\n")
