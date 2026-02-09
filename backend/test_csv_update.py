import os
import pandas as pd
from datetime import datetime

# Path to your CSV (same as in auto_update_and_push.py)
CSV_PATH = r"C:\Users\HomePC\Documents\PROJECTS\NSE_STOCK_APP\data\processed\NSE_20_stocks_2013_2025_features_target.csv"

# Ensure folder exists
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# Check if CSV exists
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(columns=["Stock", "Price", "Timestamp"])

# Append a dummy row
df = pd.concat([df, pd.DataFrame([{
    "Stock": "TEST",
    "Price": 0,
    "Timestamp": datetime.now()
}])], ignore_index=True)

# Save back to CSV
df.to_csv(CSV_PATH, index=False)

print(f"✅ Dummy row added to CSV. Run auto_update_and_push.py to test commit & push.")
