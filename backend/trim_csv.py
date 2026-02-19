import pandas as pd

# Path to your existing CSV
csv_path = r"C:\Users\HomePC\Documents\PROJECTS\NSE_STOCK_APP\data\processed\NSE_20_stocks_2013_2025_features_target.csv"

# Load CSV
df = pd.read_csv(csv_path, low_memory=False)

# Ensure date column is datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Keep only rows from 2022 onwards
df = df[df['date'] >= '2022-01-01']

# Save to a new CSV (don't overwrite original yet)
trimmed_csv_path = r"C:\Users\HomePC\Documents\PROJECTS\NSE_STOCK_APP\data\processed\NSE_20_stocks_2022_2026.csv"
df.to_csv(trimmed_csv_path, index=False)

print(f"Trimmed CSV saved: {trimmed_csv_path}, total rows: {len(df)}")
