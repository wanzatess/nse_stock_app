"""
NSE Stock Prediction Service (MODEL ONLY)
This service ONLY handles ML predictions to keep memory usage separate
Deploy this as a second Render service
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
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
model = None

# ------------------------------
# PATHS
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "stock_model_compressed.pkl")
HF_MODEL_URL = "https://huggingface.co/wanzatess/nse_stock_model/resolve/main/stock_model_compressed.pkl"

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def load_pickle_file(filepath):
    """Intelligently load a pickle file"""
    print(f"📂 Attempting to load: {filepath}")
    
    with open(filepath, 'rb') as f:
        first_bytes = f.read(10)
    
    print(f"🔍 File signature: {first_bytes[:4].hex()}")
    
    # Try joblib.load first
    print(f"🔧 Trying joblib.load...")
    try:
        loaded_model = joblib.load(filepath)
        print(f"✅ Loaded with joblib.load")
        return loaded_model
    except Exception as e:
        print(f"⚠️ joblib failed: {e}")

    # Try regular pickle
    print(f"📦 Trying regular pickle...")
    try:
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"✅ Loaded with regular pickle")
        return loaded_model
    except Exception as e:
        print(f"⚠️ regular pickle failed: {e}")
    
    raise Exception("All loading methods failed")


def download_model():
    """Download the model from HuggingFace if not present"""
    global model
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"📦 Found existing model file: {MODEL_PATH}")
        print(f"📊 File size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
        try:
            loaded_model = load_pickle_file(MODEL_PATH)
            print(f"✅ Model loaded successfully!")
            print(f"🔍 Model type: {type(loaded_model)}")
            
            if hasattr(loaded_model, 'predict'):
                print(f"✅ Model has predict method")
            if hasattr(loaded_model, 'predict_proba'):
                print(f"✅ Model has predict_proba method")
            
            model = loaded_model
            print(f"✅ Global model variable set successfully")
            return
        except Exception as e:
            print(f"⚠️ Failed to load existing model: {e}")
            print("🗑️ Deleting corrupted file and re-downloading...")
            try:
                os.remove(MODEL_PATH)
            except:
                pass
    
    # Download model from HuggingFace
    print(f"⬇️ Downloading model from HuggingFace...")
    print(f"🔗 URL: {HF_MODEL_URL}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/octet-stream'
        }
        
        response = requests.get(HF_MODEL_URL, headers=headers, timeout=120, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        print(f"📋 Content-Type: {content_type}")
        
        if 'text/html' in content_type:
            print(f"❌ Received HTML instead of binary file")
            print(f"💡 Trying with ?download=true parameter...")
            response = requests.get(f"{HF_MODEL_URL}?download=true", headers=headers, timeout=120, stream=True)
            response.raise_for_status()
        
        print(f"💾 Saving model to: {MODEL_PATH}")
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model downloaded successfully!")
        print(f"📊 File size: {file_size / 1024 / 1024:.2f} MB")
        
        loaded_model = load_pickle_file(MODEL_PATH)
        print(f"✅ Model loaded successfully!")
        print(f"🔍 Model type: {type(loaded_model)}")
        
        if hasattr(loaded_model, 'predict'):
            print(f"✅ Model has predict method")
        if hasattr(loaded_model, 'predict_proba'):
            print(f"✅ Model has predict_proba method")
        
        model = loaded_model
        print(f"✅ Global model variable set successfully")
        
    except Exception as e:
        print(f"❌ Failed to download/load model: {e}")
        print(f"⚠️ Predictions will not be available")
        model = None


def predict_with_model(features: list):
    """Make prediction using the loaded model"""
    if model is None:
        print("⚠️ Model not loaded, attempting to load...")
        try:
            download_model()
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    if model is None:
        print("❌ Model still not available after loading attempt")
        return None
    
    try:
        # Convert features to numpy array with shape (1, n_features)
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)
        raw_pred = prediction[0]

        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)
            conf = float(max(probabilities[0]))
            
            try:
                pred_value = int(raw_pred)
            except Exception:
                try:
                    pred_value = float(raw_pred)
                except Exception:
                    pred_value = str(raw_pred)

            return {
                "prediction": pred_value,
                "confidence": conf,
                "probabilities": probabilities[0].tolist()
            }
        else:
            try:
                pred_value = float(raw_pred)
            except Exception:
                pred_value = str(raw_pred)

            return {
                "prediction": pred_value,
                "confidence": None,
                "probabilities": None
            }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None


# ------------------------------
# REQUEST MODELS
# ------------------------------
class PredictRequest(BaseModel):
    features: list  # [day_price, ma_5, ma_10, pct_from_12m_low, pct_from_12m_high, daily_return, daily_volatility]
    symbol: str = None  # Optional, for logging


# ------------------------------
# API ROUTES
# ------------------------------
@app.get("/")
def read_root():
    return {
        "service": "NSE Stock Prediction Service",
        "status": "active",
        "model_loaded": model is not None
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
    }


@app.get("/model-status")
def model_status():
    """Detailed model status"""
    status = {
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }
    if model is None:
        status["hint"] = "Model not loaded. POST to /model-load to attempt loading now."
    else:
        try:
            status["model_type"] = str(type(model))
        except Exception:
            status["model_type"] = "unknown"
    return status


@app.post("/model-load")
def model_load():
    """Trigger model download/load on demand"""
    if model is not None:
        return {"ok": True, "message": "Model already loaded"}
    try:
        download_model()
        if model is None:
            raise Exception("Model failed to load")
        return {"ok": True, "message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict stock movement based on features
    
    Features should be in this order:
    [day_price, ma_5, ma_10, pct_from_12m_low, pct_from_12m_high, daily_return, daily_volatility]
    """
    if len(request.features) != 7:
        raise HTTPException(
            status_code=400, 
            detail="Expected 7 features: [day_price, ma_5, ma_10, pct_from_12m_low, pct_from_12m_high, daily_return, daily_volatility]"
        )
    
    prediction_result = predict_with_model(request.features)
    
    if prediction_result is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available or prediction failed. Please try again later."
        )
    
    response = {
        "prediction": prediction_result,
        "features_used": request.features
    }
    
    if request.symbol:
        response["symbol"] = request.symbol
    
    return response


# ------------------------------
# STARTUP
# ------------------------------
@app.on_event("startup")
async def startup_event():
    print("\n🚀 NSE PREDICTION SERVICE - STARTING UP")
    print("🤖 This service ONLY handles ML predictions")
    print("💾 Loading model into memory...")
    
    try:
        download_model()
        if model is not None:
            print("✅ Model loaded successfully and ready for predictions")
        else:
            print("⚠️ Model not loaded - will attempt on first prediction")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("⚠️ Will attempt to load on first prediction request")
    
    print("✅ Prediction service ready!\n")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 NSE PREDICTION SERVICE - STARTING UP")
    download_model()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))