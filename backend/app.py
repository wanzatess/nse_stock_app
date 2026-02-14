# backend/app.py
import math
import pandas as pd
import joblib
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# --------------------------
# CONFIG
# --------------------------
DATA_PATH = Path("src/data/processed/NSE_20_stocks_2013_2025_features_target.csv")
MODEL_PATH = Path("src/models/stock_model_compressed.pkl")

# --------------------------
# INIT APP
# --------------------------
app = FastAPI(title="NSE Stock Prediction API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# LOAD DATA
# --------------------------
print("🔄 Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded! {len(df)} records, {df['symbol'].nunique()} stocks")

# Lazy-load model
model = None

def load_model():
    global model
    if model is None:
        print("⬇️ Loading model...")
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded!")
    return model

# --------------------------
# UTILS
# --------------------------
def sanitize_stock_data(stock_list):
    """Replace NaN/inf floats with 0.0 to make JSON compliant."""
    for stock in stock_list:
        for key, value in stock.items():
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    stock[key] = 0.0
    return stock_list

def top_n_stocks(criteria: str, n: int = 20):
    if criteria not in df.columns:
        return []
    sorted_df = df.sort_values(by=criteria, ascending=(criteria == "losers"))
    top_list = sorted_df.head(n).to_dict(orient="records")
    return sanitize_stock_data(top_list)

# --------------------------
# ROUTES
# --------------------------
@app.get("/")
async def root():
    return {"message": "NSE Stock Prediction API is live!"}

@app.get("/stocks")
async def get_stocks():
    stocks = df["symbol"].unique().tolist()
    return JSONResponse(content=stocks)

@app.get("/market-overview")
async def market_overview():
    total_stocks = df['symbol'].nunique()
    gainers = len(df[df['change'] > 0])
    losers = len(df[df['change'] < 0])
    unchanged = total_stocks - gainers - losers
    average_change = df['change'].mean()
    total_volume = df['volume'].sum()
    overview = {
        "total_stocks": total_stocks,
        "gainers": gainers,
        "losers": losers,
        "unchanged": unchanged,
        "average_change": average_change if not math.isnan(average_change) else 0.0,
        "total_volume": total_volume,
        "last_updated": df['date'].max() if 'date' in df.columns else None
    }
    return JSONResponse(content=overview)

@app.get("/top-stocks")
async def get_top_stocks(
    criteria: str = Query(..., description="volume | gainers | losers"),
    limit: int = Query(20, description="Number of stocks to return")
):
    top_list = top_n_stocks(criteria, limit)
    return JSONResponse(content=top_list)

@app.get("/predict")
async def predict_stock(symbol: str, days: int = 1):
    model = load_model()
    # Simple example: predict next `days` closing prices
    # Replace with your actual model input/output logic
    if symbol not in df['symbol'].values:
        return JSONResponse(content={"error": "Symbol not found"}, status_code=404)

    last_row = df[df['symbol'] == symbol].iloc[-1]
    prediction = [float(last_row['close']) * (1 + 0.01 * i) for i in range(1, days + 1)]
    prediction = [0.0 if math.isnan(p) or math.isinf(p) else p for p in prediction]
    return JSONResponse(content={"symbol": symbol, "prediction": prediction})

@app.get("/trends/{symbol}")
async def stock_trends(symbol: str, days: int = 30):
    if symbol not in df['symbol'].values:
        return JSONResponse(content={"error": "Symbol not found"}, status_code=404)
    trend_data = df[df['symbol'] == symbol].tail(days)[['date', 'close']].to_dict(orient="records")
    return JSONResponse(content=sanitize_stock_data(trend_data))
