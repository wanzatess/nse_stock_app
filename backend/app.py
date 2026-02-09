from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional, List
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from pathlib import Path

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data and model
df = None
model = None
last_modified = None

# ============================================================================
# CORRECTED FILE PATHS FOR YOUR PROJECT STRUCTURE
# ============================================================================
# Your structure: NSE_STOCK_APP/backend/app.py and NSE_STOCK_APP/data/processed/file.csv
# So from backend/app.py, the data file is at: ../data/processed/

# Get the absolute path to the project root
# app.py is in backend/, so go up one level to get to NSE_STOCK_APP/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct data path relative to project root
data_path = os.path.join(BASE_DIR, "data", "processed", "NSE_20_stocks_2013_2025_features_target.csv")

print(f"📁 Project structure:")
print(f"   Base directory: {BASE_DIR}")
print(f"   Data path: {data_path}")
print(f"   Data file exists: {os.path.exists(data_path)}")

# Model setup with Render Persistent Disk support
PERSISTENT_DISK = os.environ.get('PERSISTENT_DISK_PATH', os.path.join(BASE_DIR, 'backend', 'models'))
MODEL_FILENAME = "stock_model.pkl"
MODEL_PATH = os.path.join(PERSISTENT_DISK, MODEL_FILENAME)

# Google Drive configuration
GOOGLE_DRIVE_FILE_ID = os.environ.get('GOOGLE_DRIVE_FILE_ID', '1ewS_wYEWWxiJKu7JNeanXIsJwbR4ggt1')

print(f"📦 Model configuration:")
print(f"   Persistent disk: {PERSISTENT_DISK}")
print(f"   Model path: {MODEL_PATH}")

# Cache for live data (15 minute cache)
live_data_cache = {}
cache_timestamps = {}
CACHE_DURATION = 900  # 15 minutes in seconds

def ensure_model_directory():
    """Create model directory if it doesn't exist"""
    try:
        Path(PERSISTENT_DISK).mkdir(parents=True, exist_ok=True)
        print(f"✅ Model directory ready: {PERSISTENT_DISK}")
    except Exception as e:
        print(f"⚠️ Could not create model directory: {e}")

def download_from_google_drive(file_id, destination):
    """Download a file from Google Drive"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        total_size = 0
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
                    if total_size % (10 * 1024 * 1024) == 0:
                        print(f"   Downloaded: {total_size / (1024 * 1024):.1f} MB...")

    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True, timeout=300)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True, timeout=300)

    save_response_content(response, destination)

def download_model_to_persistent_disk():
    """Download model to persistent disk (one-time operation)"""
    
    if os.path.exists(MODEL_PATH):
        file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model found on persistent disk ({file_size_mb:.2f} MB)")
        return True
    
    print("📥 Model not found. Starting download...")
    print("⏳ This may take a few minutes for large models...")
    
    try:
        # Try gdown first
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
            if os.path.exists(MODEL_PATH):
                file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
                print(f"✅ Model downloaded successfully ({file_size_mb:.2f} MB)")
                return True
            
        except ImportError:
            print("⚠️ gdown not installed, using fallback...")
        
        # Fallback
        download_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
        
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"✅ Model downloaded ({file_size_mb:.2f} MB)")
            return True
        else:
            raise Exception("Downloaded file is empty or missing")
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("   App will continue without model (predictions unavailable)")
        return False

def load_model():
    """Load model from persistent disk"""
    global model
    
    if model is not None:
        return True
    
    ensure_model_directory()
    
    # Download if needed
    if not download_model_to_persistent_disk():
        return False
    
    print(f"📦 Loading model from {MODEL_PATH}...")
    
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        if not hasattr(model, 'predict'):
            raise Exception("Model doesn't have 'predict' method")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Standard load failed: {e}")
        
        try:
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded with pickle!")
            return True
        except Exception as e2:
            print(f"❌ All loading methods failed: {e2}")
            model = None
            return False

def reload_data_if_needed():
    global df, last_modified
    try:
        if not os.path.exists(data_path):
            print(f"❌ Data file not found at: {data_path}")
            return
        
        current_modified = os.path.getmtime(data_path)
        if last_modified is None or current_modified > last_modified:
            print(f"📄 Loading data from: {data_path}")
            df = pd.read_csv(data_path, low_memory=False)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            last_modified = current_modified
            print(f"✅ Data loaded! {len(df)} records, {len(df['code'].unique())} stocks")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        if df is None:
            raise

# Initialize on startup
print("\n" + "=" * 70)
print("🚀 NSE STOCK PREDICTION API - STARTING UP")
print("=" * 70)

reload_data_if_needed()
model_loaded = load_model()

print("=" * 70)
print(f"Status: {'✅ READY' if (df is not None and model_loaded) else '⚠️ DEGRADED'}")
print(f"Data: {'✅ Loaded' if df is not None else '❌ Not loaded'}")
print(f"Model: {'✅ Loaded' if model_loaded else '⚠️ Not loaded (predictions disabled)'}")
print("=" * 70 + "\n")

def fetch_live_stock_data(symbol):
    now = datetime.now()
    if symbol in cache_timestamps:
        age = (now - cache_timestamps[symbol]).seconds
        if age < CACHE_DURATION:
            return live_data_cache[symbol]
    try:
        url = f"https://afx.kwayisi.org/nse/{symbol.lower()}.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = pd.read_html(response.text)
            if tables:
                data = {
                    'symbol': symbol.upper(),
                    'last_updated': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'data_source': 'live_afx'
                }
                live_data_cache[symbol] = data
                cache_timestamps[symbol] = now
                return data
        return None
    except Exception as e:
        print(f"Error fetching live data for {symbol}: {e}")
        return None

def get_latest_stock_data(symbol):
    reload_data_if_needed()
    if df is None:
        return None, None
    
    stock_df = df[df['code'] == symbol].sort_values('date', ascending=False)
    if stock_df.empty:
        return None, None
    
    row = stock_df.iloc[0]
    change_pct = row['changepct']
    if isinstance(change_pct, str):
        change_pct = float(change_pct.replace('%', ''))
    
    return {
        'symbol': symbol,
        'current_price': float(row['day_price']),
        'previous_price': float(row['previous']) if pd.notna(row['previous']) else float(row['day_price']),
        'change': float(row['change']) if pd.notna(row['change']) else 0.0,
        'change_percent': float(change_pct) if pd.notna(change_pct) else 0.0,
        'day_low': float(row['day_low']) if pd.notna(row['day_low']) else float(row['day_price']),
        'day_high': float(row['day_high']) if pd.notna(row['day_high']) else float(row['day_price']),
        'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
        '12m_low': float(row['12m_low']) if pd.notna(row['12m_low']) else float(row['day_price']),
        '12m_high': float(row['12m_high']) if pd.notna(row['12m_high']) else float(row['day_price']),
        'name': row['name'],
        'ma_5': float(row['ma_5']) if pd.notna(row['ma_5']) else float(row['day_price']),
        'ma_10': float(row['ma_10']) if pd.notna(row['ma_10']) else float(row['day_price']),
        'pct_from_12m_low': float(row['pct_from_12m_low']) if pd.notna(row['pct_from_12m_low']) else 0.0,
        'pct_from_12m_high': float(row['pct_from_12m_high']) if pd.notna(row['pct_from_12m_high']) else 0.0,
        'daily_return': float(row['daily_return']) if pd.notna(row['daily_return']) else 0.0,
        'daily_volatility': float(row['daily_volatility']) if pd.notna(row['daily_volatility']) else 0.0,
        'last_updated': row['date'].strftime('%Y-%m-%d'),
        'data_source': 'database'
    }, 'database'

class PredictRequest(BaseModel):
    symbol: str

@app.get("/")
def root():
    reload_data_if_needed()
    return {
        "status": "healthy" if (model is not None and df is not None) else "degraded",
        "message": "NSE Kenya Stock Prediction API",
        "model_status": "loaded" if model is not None else "not_loaded",
        "data_status": "loaded" if df is not None else "not_loaded",
        "total_stocks": len(df['code'].unique()) if df is not None else 0,
        "total_records": len(df) if df is not None else 0,
        "last_data_update": datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S') if last_modified else "Unknown"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": df is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stocks")
def get_available_stocks():
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not available")
    
    stocks_info = df.groupby('code').agg({'name': 'first'}).reset_index()
    stocks_list = [{"code": row['code'], "name": row['name']} for _, row in stocks_info.iterrows()]
    return {
        "stocks": sorted([s['code'] for s in stocks_list]),
        "stocks_with_names": sorted(stocks_list, key=lambda x: x['code']),
        "count": len(stocks_list)
    }

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Predictions are currently unavailable."
        )
    
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not available")
    
    symbol = request.symbol.upper()
    stock_data, data_source = get_latest_stock_data(symbol)
    
    if not stock_data:
        raise HTTPException(status_code=404, detail=f"No data found for symbol '{symbol}'")
    
    features = [[
        float(stock_data['current_price']),
        float(stock_data['ma_5']),
        float(stock_data['ma_10']),
        float(stock_data['pct_from_12m_low']),
        float(stock_data['pct_from_12m_high']),
        float(stock_data['daily_return']),
        float(stock_data['daily_volatility'])
    ]]
    
    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    return {
        "symbol": symbol,
        "name": stock_data['name'],
        "prediction": prediction,
        "current_price": stock_data['current_price'],
        "previous_price": stock_data['previous_price'],
        "change": stock_data['change'],
        "change_percent": stock_data['change_percent'],
        "day_low": stock_data['day_low'],
        "day_high": stock_data['day_high'],
        "12m_low": stock_data['12m_low'],
        "12m_high": stock_data['12m_high'],
        "volume": stock_data['volume'],
        "last_updated": stock_data['last_updated'],
        "data_source": data_source,
        "features_used": {
            "ma_5": stock_data['ma_5'],
            "ma_10": stock_data['ma_10'],
            "pct_from_12m_low": stock_data['pct_from_12m_low'],
            "pct_from_12m_high": stock_data['pct_from_12m_high'],
            "daily_return": stock_data['daily_return'],
            "daily_volatility": stock_data['daily_volatility']
        }
    }

@app.get("/top-stocks")
def get_top_stocks(criteria: str = "gainers", limit: int = 10):
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not available")
    
    latest_data = df.sort_values('date', ascending=False).groupby('code').first().reset_index()
    required_cols = ['day_price', 'ma_5', 'ma_10', 'pct_from_12m_low', 'pct_from_12m_high', 'daily_return', 'daily_volatility']
    latest_data = latest_data.dropna(subset=required_cols)
    
    if criteria == "gainers":
        result = latest_data.sort_values('daily_return', ascending=False).head(limit)
    elif criteria == "losers":
        result = latest_data.sort_values('daily_return', ascending=True).head(limit)
    elif criteria == "volume":
        result = latest_data.sort_values('volume', ascending=False).head(limit)
    elif criteria == "buy_signals":
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")
        
        predictions = []
        for _, row in latest_data.iterrows():
            features = [[
                float(row['day_price']),
                float(row['ma_5']),
                float(row['ma_10']),
                float(row['pct_from_12m_low']),
                float(row['pct_from_12m_high']),
                float(row['daily_return']),
                float(row['daily_volatility'])
            ]]
            try:
                pred = model.predict(features)[0]
                if pred.lower() == 'buy':
                    predictions.append(row)
            except:
                continue
        result = pd.DataFrame(predictions).head(limit) if predictions else pd.DataFrame()
    else:
        raise HTTPException(status_code=400, detail="Invalid criteria")
    
    stocks = []
    for _, row in result.iterrows():
        change_pct = row['changepct']
        if isinstance(change_pct, str):
            change_pct = float(change_pct.replace('%', ''))
        stocks.append({
            "symbol": row['code'],
            "name": row['name'],
            "current_price": float(row['day_price']),
            "change": float(row['change']) if pd.notna(row['change']) else 0.0,
            "change_percent": float(change_pct) if pd.notna(change_pct) else 0.0,
            "volume": int(row['volume']) if not pd.isna(row['volume']) else 0,
        })
    
    return {"criteria": criteria, "stocks": stocks, "count": len(stocks)}

@app.get("/market-overview")
def get_market_overview():
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not available")
    
    latest_data = df.sort_values('date', ascending=False).groupby('code').first().reset_index()
    total_stocks = len(latest_data)
    gainers = len(latest_data[latest_data['daily_return'] > 0])
    losers = len(latest_data[latest_data['daily_return'] < 0])
    unchanged = total_stocks - gainers - losers
    avg_change = latest_data['daily_return'].mean()
    total_volume = latest_data['volume'].sum()
    
    return {
        "total_stocks": total_stocks,
        "gainers": gainers,
        "losers": losers,
        "unchanged": unchanged,
        "average_change": float(avg_change),
        "total_volume": int(total_volume) if not pd.isna(total_volume) else 0,
        "last_updated": latest_data['date'].max().strftime('%Y-%m-%d')
    }

@app.get("/trends/{symbol}")
def get_stock_trends(symbol: str, days: int = 30):
    reload_data_if_needed()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not available")
    
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
        raise HTTPException(status_code=503, detail="Data not available")
    
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
    
    return {
        "symbol": symbol,
        "name": stock_df.iloc[0]['name'],
        "data": history
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)