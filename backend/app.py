from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from typing import Optional, List
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

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
data_path = "../data/processed/NSE_20_stocks_2013_2025_features_target.csv"

# Model setup
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "stock_model.pkl")

# Updated Google Drive ID (from your gdown attempt in the original code)
GOOGLE_DRIVE_FILE_ID = "1ewS_wYEWWxiJKu7JNeanXIsJwbR4ggt1"

# Cache for live data (15 minute cache)
live_data_cache = {}
cache_timestamps = {}
CACHE_DURATION = 900  # 15 minutes in seconds

def reload_data_if_needed():
    global df, last_modified
    try:
        current_modified = os.path.getmtime(data_path)
        if last_modified is None or current_modified > last_modified:
            print(f"📄 Reloading data... (last modified: {datetime.fromtimestamp(current_modified)})")
            df = pd.read_csv(data_path, low_memory=False)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            last_modified = current_modified
            print(f"✅ Data reloaded! {len(df)} records")
    except Exception as e:
        print(f"⚠️ Error reloading data: {e}")
        if df is None:
            df = pd.read_csv(data_path, low_memory=False)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

def download_from_google_drive(file_id, destination):
    """Download a file from Google Drive using direct download link"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def download_model_if_needed():
    """Download the model from Google Drive if it doesn't exist locally"""
    global MODEL_PATH
    
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model found at {MODEL_PATH}")
        return
    
    print("📥 Model not found locally. Downloading from Google Drive...")
    
    try:
        # Try using gdown first (more reliable for large files)
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            print("✅ Model downloaded successfully using gdown")
            return
        except ImportError:
            print("⚠️ gdown not available, using fallback method...")
        
        # Fallback to manual download
        download_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
        
        # Verify the download
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            print(f"✅ Model downloaded successfully ({os.path.getsize(MODEL_PATH)} bytes)")
        else:
            raise Exception("Downloaded file is empty or doesn't exist")
            
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to download model file: {str(e)}"
        )

def load_model():
    """Load the machine learning model with enhanced error handling"""
    global model
    
    if model is not None:
        print("✅ Model already loaded")
        return
    
    try:
        # Ensure model file exists
        download_model_if_needed()
        
        print(f"📦 Loading model from {MODEL_PATH}...")
        
        # Try different loading methods
        try:
            # Method 1: Standard joblib load
            model = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully with joblib!")
            
        except Exception as e1:
            print(f"⚠️ joblib.load failed: {e1}")
            print("🔄 Trying alternative loading method...")
            
            try:
                # Method 2: Use pickle directly with different protocols
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                print("✅ Model loaded successfully with pickle!")
                
            except Exception as e2:
                print(f"⚠️ pickle.load failed: {e2}")
                print("🔄 Trying with pickle protocol 4...")
                
                try:
                    # Method 3: Force pickle protocol 4
                    with open(MODEL_PATH, 'rb') as f:
                        model = pickle.load(f, encoding='latin1')
                    print("✅ Model loaded successfully with pickle (latin1)!")
                    
                except Exception as e3:
                    error_msg = f"""
Failed to load model after multiple attempts:
- joblib error: {str(e1)}
- pickle error: {str(e2)}
- pickle latin1 error: {str(e3)}

This usually indicates:
1. Python version mismatch between training and deployment
2. Different scikit-learn versions
3. Corrupted model file
4. Incompatible pickle protocol

Solutions:
1. Retrain and save the model in the deployment environment
2. Use joblib with explicit protocol: joblib.dump(model, 'model.pkl', protocol=4)
3. Check scikit-learn versions match
"""
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg)
        
        # Verify the model has a predict method
        if not hasattr(model, 'predict'):
            raise Exception("Loaded object doesn't have a 'predict' method - may not be a valid model")
            
        print("✅ Model validation successful!")
        
    except Exception as e:
        print(f"❌ Critical error loading model: {e}")
        # Don't raise here - allow the app to start without the model
        # Predictions will fail gracefully
        model = None

# Load model and data on startup
try:
    load_model()
    reload_data_if_needed()
except Exception as e:
    print(f"⚠️ Startup warning: {e}")
    print("⚠️ App will start but predictions may not work until model is available")

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
        "status": "healthy" if model is not None else "degraded",
        "message": "NSE Kenya Stock Prediction API",
        "model_status": "loaded" if model is not None else "not_loaded",
        "data_mode": "Auto-reloading (updates when CSV changes)",
        "total_stocks": len(df['code'].unique()) if df is not None else 0,
        "total_records": len(df) if df is not None else 0,
        "last_data_update": datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S') if last_modified else "Unknown"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for deployment monitoring"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "data_loaded": df is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stocks")
def get_available_stocks():
    reload_data_if_needed()
    stocks_info = df.groupby('code').agg({'name': 'first'}).reset_index()
    stocks_list = [{"code": row['code'], "name": row['name']} for _, row in stocks_info.iterrows()]
    return {
        "stocks": sorted([s['code'] for s in stocks_list]),
        "stocks_with_names": sorted(stocks_list, key=lambda x: x['code']),
        "count": len(stocks_list)
    }

@app.post("/predict")
def predict(request: PredictRequest):
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please check server logs or contact administrator."
        )
    
    reload_data_if_needed()
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
            raise HTTPException(status_code=503, detail="Model not available for predictions")
        
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