from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from datetime import datetime
import requests
import pickle
import joblib

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
model = None

# ------------------------------
# PATHS
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up one level from backend/
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "NSE_20_stocks_2013_2025_features_target.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "stock_model_compressed.pkl")

# HuggingFace Model URL
HF_MODEL_URL = "https://huggingface.co/wanzatess/nse_stock_model/resolve/main/stock_model_compressed.pkl"

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def download_model():
    """Download the model from HuggingFace if not present"""
    global model
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"📦 Loading existing model from: {MODEL_PATH}")
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Model loaded successfully!")
            return
        except Exception as e:
            print(f"⚠️ Failed to load existing model: {e}")
            print("🔄 Re-downloading model...")
    
    # Download model from HuggingFace with proper headers
    print(f"⬇️ Downloading model from HuggingFace...")
    try:
        # Add headers to ensure we get the raw file, not HTML
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/octet-stream'
        }
        
        response = requests.get(HF_MODEL_URL, headers=headers, timeout=120, stream=True)
        response.raise_for_status()
        
        # Check if we got HTML instead of binary data
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print(f"❌ Received HTML instead of binary file. Content-Type: {content_type}")
            print(f"💡 Try accessing the URL directly: {HF_MODEL_URL}?download=true")
            # Try with download parameter
            response = requests.get(f"{HF_MODEL_URL}?download=true", headers=headers, timeout=120, stream=True)
            response.raise_for_status()
        
        # Save the model
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Model downloaded and saved to: {MODEL_PATH}")
        print(f"📊 File size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
        
        # Load the model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded successfully!")
        print(f"🔍 Model type: {type(model)}")
        
    except pickle.UnpicklingError as e:
        print(f"❌ Failed to unpickle model: {e}")
        print(f"🔍 File might be corrupted or not a valid pickle file")
        # Try to read first few bytes to check
        try:
            with open(MODEL_PATH, 'rb') as f:
                first_bytes = f.read(100)
                print(f"🔍 First bytes: {first_bytes[:50]}")
        except:
            pass
        model = None
    except Exception as e:
        print(f"❌ Failed to download/load model: {e}")
        print(f"⚠️ Predictions will not be available")
        model = None


def reload_data_if_needed():
    global df, last_modified
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ Data file not found: {DATA_PATH}")
        return

    current_modified = os.path.getmtime(DATA_PATH)
    if last_modified is None or current_modified > last_modified:
        print(f"🔄 Loading data from: {DATA_PATH}")
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


def predict_with_model(features: list):
    """Make prediction using the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later."
        )
    
    try:
        # Convert features to numpy array with shape (1, n_features)
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)
            
            # Assuming binary or multi-class classification
            # Adjust based on your model's output
            return {
                "prediction": int(prediction[0]),
                "confidence": float(max(probabilities[0])),
                "probabilities": probabilities[0].tolist()
            }
        else:
            return {
                "prediction": float(prediction[0])
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


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
        "model_loaded": model is not None,
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

    features = [
        stock["current_price"],
        stock["ma_5"],
        stock["ma_10"],
        stock["pct_from_12m_low"],
        stock["pct_from_12m_high"],
        stock["daily_return"],
        stock["daily_volatility"]
    ]

    prediction_result = predict_with_model(features)
    
    return {
        "symbol": symbol,
        "name": stock["name"],
        "prediction": prediction_result,
        **stock
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
        "average_change": float(avg_change),
        "total_volume": int(total_volume),
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
        # Get 'buy' signals via model predictions
        predictions = []
        for _, row in latest_data.iterrows():
            features = [
                row['day_price'], row['ma_5'], row['ma_10'],
                row['pct_from_12m_low'], row['pct_from_12m_high'],
                row['daily_return'], row['daily_volatility']
            ]
            try:
                pred = predict_with_model(features)
                # Adjust this logic based on your model's output
                # Assuming prediction > 0 means buy signal
                if pred and pred.get('prediction', 0) > 0:
                    predictions.append(row)
            except:
                continue
        result = pd.DataFrame(predictions).head(limit) if predictions else pd.DataFrame()
    else:
        raise HTTPException(status_code=400, detail="Invalid criteria")

    stocks = []
    for _, row in result.iterrows():
        change_pct = row["changepct"]
        if isinstance(change_pct, str):
            change_pct = float(change_pct.replace("%", ""))
        stocks.append({
            "symbol": row["code"],
            "name": row["name"],
            "current_price": float(row["day_price"]),
            "change": float(row["change"]) if pd.notna(row["change"]) else 0.0,
            "change_percent": float(change_pct),
            "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0
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
        "highest_price": float(stock_df['day_high'].max()),
        "lowest_price": float(stock_df['day_low'].min()),
        "average_price": float(stock_df['day_price'].mean()),
        "current_price": float(stock_df.iloc[0]['day_price']),
        "price_change": float(stock_df.iloc[0]['day_price'] - stock_df.iloc[-1]['day_price']),
        "average_volume": int(stock_df['volume'].mean()) if not pd.isna(stock_df['volume'].mean()) else 0
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
            change_pct = float(change_pct.replace('%', ''))
        history.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "price": float(row['day_price']),
            "low": float(row['day_low']) if pd.notna(row['day_low']) else float(row['day_price']),
            "high": float(row['day_high']) if pd.notna(row['day_high']) else float(row['day_price']),
            "volume": int(row['volume']) if not pd.isna(row['volume']) else 0,
            "change_percent": float(change_pct) if pd.notna(change_pct) else 0.0
        })

    return {"symbol": symbol, "name": stock_df.iloc[0]['name'], "data": history}


# ------------------------------
# STARTUP
# ------------------------------
@app.on_event("startup")
async def startup_event():
    print("\n🚀 NSE STOCK PREDICTION API - STARTING UP")
    reload_data_if_needed()
    download_model()
    print("✅ Startup complete!\n")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 NSE STOCK PREDICTION API - STARTING UP")
    reload_data_if_needed()
    download_model()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))