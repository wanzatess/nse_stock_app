"""
NSE Stock Data Service (NO MODEL - Memory Optimized)
This service handles all data endpoints WITHOUT loading the ML model
Deploy this as your main service on Render
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import requests

# ------------------------------
# FASTAPI APP
# ------------------------------
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
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "NSE_20_stocks_2013_2025_features_target.csv")

# URL of your separate prediction service
PREDICTION_SERVICE_URL = "https://nse-predictions.onrender.com"  # You'll create this

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def sanitize_float(value, default=0.0):
    """Convert value to float and handle NaN, inf, -inf"""
    try:
        if pd.isna(value):
            return default
        float_val = float(value)
        if not np.isfinite(float_val):
            return default
        return float_val
    except (ValueError, TypeError, OverflowError):
        return default


def reload_data_if_needed():
    global df, last_modified
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ Data file not found: {DATA_PATH}")
        return

    current_modified = os.path.getmtime(DATA_PATH)
    if last_modified is None or current_modified > last_modified:
        print(f"🔄 Loading data from: {DATA_PATH}")
        
        # Load essential columns only
        essential_cols = ['date', 'code', 'name', 'day_price', 'previous', 'change', 
                        'changepct', 'volume', 'day_low', 'day_high', 'ma_5', 'ma_10',
                        'pct_from_12m_low', 'pct_from_12m_high', 'daily_return', 'daily_volatility']
        
        df = pd.read_csv(DATA_PATH, low_memory=False, usecols=essential_cols)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # MEMORY OPTIMIZATION: Keep only recent years
        # Change this cutoff date to load more/less data
        # Options: 
        # - 2018-01-01 = 7 years (~120MB)
        # - 2020-01-01 = 5 years (~90MB)
        # - 2022-01-01 = 3 years (~60MB)
        CUTOFF_DATE = pd.Timestamp('2018-01-01')
        
        original_rows = len(df)
        df = df[df["date"] >= CUTOFF_DATE]
        filtered_rows = len(df)
        
        print(f"📅 Filtered data: {original_rows:,} rows → {filtered_rows:,} rows (from {CUTOFF_DATE.strftime('%Y-%m-%d')})")
        
        # Optimize data types to use less memory
        df['code'] = df['code'].astype('category')
        df['name'] = df['name'].astype('category')
        
        # Convert float64 to float32 to save ~50% memory on floats
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        last_modified = current_modified
        
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"✅ Data loaded successfully")
        print(f"📊 Stocks: {df['code'].nunique()}, Records: {len(df):,}, Memory: {memory_mb:.2f} MB")
        print(f"📆 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")


# ------------------------------
# API ROUTES - DATA ONLY (NO PREDICTIONS)
# ------------------------------

@app.get("/")
def read_root():
    return {
        "service": "NSE Stock Data Service",
        "status": "active",
        "note": "For predictions, use the prediction service"
    }


@app.get("/health")
def health():
    reload_data_if_needed()
    return {
        "status": "ok" if df is not None else "degraded",
        "data_loaded": df is not None,
        "total_stocks": int(df["code"].nunique()) if df is not None else 0,
    }


@app.get("/stocks")
def get_all_stocks():
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    stocks = df.groupby('code')['name'].first().reset_index()
    return {
        "stocks": [
            {"symbol": row["code"], "name": row["name"]} 
            for _, row in stocks.iterrows()
        ],
        "count": len(stocks)
    }


@app.get("/market-overview")
def get_market_overview():
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    latest_data = df.sort_values("date", ascending=False).groupby("code").first().reset_index()
    total_stocks = len(latest_data)
    gainers = len(latest_data[latest_data["daily_return"] > 0])
    losers = len(latest_data[latest_data["daily_return"] < 0])
    unchanged = total_stocks - gainers - losers
    avg_change = latest_data["daily_return"].mean()
    total_volume = latest_data["volume"].sum()

    return {
        "total_stocks": total_stocks,
        "gainers": gainers,
        "losers": losers,
        "unchanged": unchanged,
        "average_change": sanitize_float(avg_change, 0.0),
        "total_volume": int(total_volume) if np.isfinite(total_volume) else 0,
        "last_updated": latest_data["date"].max().strftime("%Y-%m-%d")
    }


@app.get("/top-stocks")
def get_top_stocks(criteria: str = "gainers", limit: int = 10):
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    latest_data = df.sort_values("date", ascending=False).groupby("code").first().reset_index()
    required_cols = ['day_price', 'ma_5', 'ma_10', 'pct_from_12m_low', 
                     'pct_from_12m_high', 'daily_return', 'daily_volatility']
    latest_data = latest_data.dropna(subset=required_cols)

    if criteria == "gainers":
        result = latest_data.sort_values('daily_return', ascending=False).head(limit)
    elif criteria == "losers":
        result = latest_data.sort_values('daily_return', ascending=True).head(limit)
    elif criteria == "volume":
        result = latest_data.sort_values('volume', ascending=False).head(limit)
    elif criteria == "buy_signals":
        # Simple heuristic (no model needed)
        predictions = []
        for _, row in latest_data.iterrows():
            if (row['daily_return'] > 0 and 
                row['day_price'] < row['ma_10'] and
                pd.notna(row['daily_return'])):
                predictions.append(row)
        result = pd.DataFrame(predictions).head(limit) if predictions else pd.DataFrame()
    else:
        raise HTTPException(status_code=400, detail="Invalid criteria")

    stocks = []
    for _, row in result.iterrows():
        change_pct = row["changepct"]
        if isinstance(change_pct, str):
            try:
                change_pct = float(change_pct.replace("%", ""))
            except:
                change_pct = 0.0
        
        stocks.append({
            "symbol": row["code"],
            "name": row["name"],
            "current_price": sanitize_float(row["day_price"], 0.0),
            "change": sanitize_float(row["change"], 0.0),
            "change_percent": sanitize_float(change_pct, 0.0),
            "volume": int(row["volume"]) if pd.notna(row["volume"]) and np.isfinite(row["volume"]) else 0
        })

    return {"criteria": criteria, "stocks": stocks, "count": len(stocks)}


@app.get("/trends/{symbol}")
def get_stock_trends(symbol: str, days: int = 30):
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    symbol = symbol.upper()
    stock_df = df[df['code'] == symbol].sort_values('date', ascending=False).head(days)
    if stock_df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    prices = stock_df['day_price'].values
    if len(prices) > 1:
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
    else:
        trend = "insufficient_data"

    return {
        "symbol": symbol,
        "name": stock_df.iloc[0]['name'],
        "trend": trend,
        "period_days": days,
        "highest_price": sanitize_float(stock_df['day_high'].max(), 0.0),
        "lowest_price": sanitize_float(stock_df['day_low'].min(), 0.0),
        "average_price": sanitize_float(stock_df['day_price'].mean(), 0.0),
        "current_price": sanitize_float(stock_df.iloc[0]['day_price'], 0.0),
        "price_change": sanitize_float(stock_df.iloc[0]['day_price'] - stock_df.iloc[-1]['day_price'], 0.0),
        "average_volume": int(stock_df['volume'].mean()) if pd.notna(stock_df['volume'].mean()) and np.isfinite(stock_df['volume'].mean()) else 0
    }


@app.get("/history/{symbol}")
def get_history(symbol: str, days: int = 30):
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    symbol = symbol.upper()
    stock_df = df[df['code'] == symbol].sort_values('date', ascending=False).head(days)
    if stock_df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    history = []
    for _, row in stock_df.iterrows():
        change_pct = row['changepct']
        if isinstance(change_pct, str):
            try:
                change_pct = float(change_pct.replace('%', ''))
            except:
                change_pct = 0.0
        
        history.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "price": sanitize_float(row['day_price'], 0.0),
            "low": sanitize_float(row['day_low'], sanitize_float(row['day_price'], 0.0)),
            "high": sanitize_float(row['day_high'], sanitize_float(row['day_price'], 0.0)),
            "volume": int(row['volume']) if pd.notna(row['volume']) and np.isfinite(row['volume']) else 0,
            "change_percent": sanitize_float(change_pct, 0.0)
        })

    return {"symbol": symbol, "name": stock_df.iloc[0]['name'], "data": history}


# ------------------------------
# PREDICTION PROXY
# ------------------------------
class PredictRequest(BaseModel):
    symbol: str


@app.post("/predict")
async def predict_proxy(request: PredictRequest):
    """
    Proxy predictions to the separate prediction service
    This allows frontend to call the data service for everything
    """
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    symbol = request.symbol.upper()
    stock_data = df[df["code"] == symbol].sort_values("date", ascending=False).head(1)
    
    if stock_data.empty:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    row = stock_data.iloc[0]
    
    # Prepare features for prediction service
    features = [
        sanitize_float(row["day_price"], 0.0),
        sanitize_float(row["ma_5"], 0.0),
        sanitize_float(row["ma_10"], 0.0),
        sanitize_float(row["pct_from_12m_low"], 0.0),
        sanitize_float(row["pct_from_12m_high"], 0.0),
        sanitize_float(row["daily_return"], 0.0),
        sanitize_float(row["daily_volatility"], 0.0)
    ]
    
    # Call prediction service
    try:
        response = requests.post(
            f"{PREDICTION_SERVICE_URL}/predict",
            json={"features": features, "symbol": symbol},
            timeout=120  # Increased to 120s for first prediction (model download)
        )
        response.raise_for_status()
        prediction_data = response.json()
        
        # Return combined data: stock info + prediction
        change_pct = row["changepct"]
        if isinstance(change_pct, str):
            try:
                change_pct = float(change_pct.replace("%", ""))
            except:
                change_pct = 0.0
        
        return {
            "symbol": symbol,
            "name": row["name"],
            "current_price": sanitize_float(row["day_price"], 0.0),
            "day_price": sanitize_float(row["day_price"], 0.0),
            "previous_price": sanitize_float(row["previous"], sanitize_float(row["day_price"], 0.0)),
            "change": sanitize_float(row["change"], 0.0),
            "change_percent": sanitize_float(change_pct, 0.0),
            "day_low": sanitize_float(row["day_low"], sanitize_float(row["day_price"], 0.0)),
            "day_high": sanitize_float(row["day_high"], sanitize_float(row["day_price"], 0.0)),
            "volume": int(row["volume"]) if pd.notna(row["volume"]) and np.isfinite(row["volume"]) else 0,
            "prediction": prediction_data["prediction"],
            "features_used": {
                "ma_5": features[1],
                "ma_10": features[2],
                "pct_from_12m_low": features[3],
                "pct_from_12m_high": features[4],
                "daily_return": features[5],
                "daily_volatility": features[6]
            }
        }
    except requests.exceptions.RequestException as e:
        # If prediction service is down, return stock data without prediction
        print(f"⚠️ Prediction service unavailable: {e}")
        
        change_pct = row["changepct"]
        if isinstance(change_pct, str):
            try:
                change_pct = float(change_pct.replace("%", ""))
            except:
                change_pct = 0.0
        
        return {
            "symbol": symbol,
            "name": row["name"],
            "current_price": sanitize_float(row["day_price"], 0.0),
            "day_price": sanitize_float(row["day_price"], 0.0),
            "previous_price": sanitize_float(row["previous"], sanitize_float(row["day_price"], 0.0)),
            "change": sanitize_float(row["change"], 0.0),
            "change_percent": sanitize_float(change_pct, 0.0),
            "day_low": sanitize_float(row["day_low"], sanitize_float(row["day_price"], 0.0)),
            "day_high": sanitize_float(row["day_high"], sanitize_float(row["day_price"], 0.0)),
            "volume": int(row["volume"]) if pd.notna(row["volume"]) and np.isfinite(row["volume"]) else 0,
            "prediction": None,
            "prediction_error": "Prediction service temporarily unavailable",
            "features_used": {
                "ma_5": features[1],
                "ma_10": features[2],
                "pct_from_12m_low": features[3],
                "pct_from_12m_high": features[4],
                "daily_return": features[5],
                "daily_volatility": features[6]
            }
        }


# ------------------------------
# STARTUP
# ------------------------------
@app.on_event("startup")
async def startup_event():
    print("\n🚀 NSE STOCK DATA SERVICE - STARTING UP")
    print("📊 Loading recent years of historical data (2018+)")
    print("💾 Memory: Optimized for Render free tier (no model)")
    reload_data_if_needed()
    print("✅ Data service ready!\n")


if __name__ == "__main__":
    import uvicorn
    reload_data_if_needed()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))