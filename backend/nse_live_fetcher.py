"""
nse_live_fetcher.py (Emoji-Free Version)
Fetches live NSE stock data and updates CSV
"""

import requests
import pandas as pd
from datetime import datetime

# ... your existing imports and code ...

def fetch_all_stocks():
    # Example structure
    sources = [
        {"name": "AFX"},
        {"name": "MyStocks"},
        # add your other sources here
    ]

    all_data = []

    for source in sources:
        # Emoji-free print
        print(f"\nTrying {source['name']}...")  

        try:
            # your fetch logic here
            data = fetch_source(source)
            all_data.append(data)
            print(f"Fetched data from {source['name']} successfully")  # was previously ✅
        except Exception as e:
            print(f"Failed to fetch from {source['name']}: {e}")      # was previously ❌

    return all_data

def fetch_source(source):
    # replace with your actual fetching code
    # for demo purposes
    return {"source": source['name'], "timestamp": datetime.now()}

def main():
    print(f"\nAuto-Update Script running at {datetime.now()}")  # was 🤖
    
    try:
        live_data = fetch_all_stocks()
        # your CSV saving logic here
        print(f"Fetched {len(live_data)} sources successfully")
    except Exception as e:
        print(f"Data fetch failed! Check your internet connection or sources: {e}")  # was ⚠️

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()
