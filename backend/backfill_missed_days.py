"""
Backfill missed NSE data by simulating past dates.
This wraps nse_live_fetcher.py and updates the main CSV.
"""

from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Import your fetcher class directly
sys.path.insert(0, r"C:\Users\HomePC\Documents\PROJECTS\NSE_STOCK_APP\backend")
from nse_live_fetcher import NSELiveDataFetcher, update_database, calculate_technical_indicators

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = r"C:\Users\HomePC\Documents\PROJECTS\NSE_STOCK_APP\data\processed\NSE_20_stocks_2013_2025_features_target.csv"

# Define the range of missed dates
# Example: missed from Feb 10 to Feb 18
START_DATE = datetime(2026, 2, 15)
END_DATE = datetime(2026, 2, 18)

# -----------------------------
# MAIN BACKFILL
# -----------------------------
def main():
    fetcher = NSELiveDataFetcher()

    # Load existing database
    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH, low_memory=False)
        existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce')
    else:
        existing_df = pd.DataFrame()

    curr_date = START_DATE
    while curr_date <= END_DATE:
        print(f"\nFetching data for {curr_date.strftime('%Y-%m-%d')}...")
        live_data = fetcher.fetch_all_stocks()
        if live_data is None or len(live_data) == 0:
            print(f"Failed to fetch data for {curr_date.strftime('%Y-%m-%d')}, skipping...")
            curr_date += timedelta(days=1)
            continue

        # Override date to simulate the missed day
        live_data['date'] = curr_date

        # Calculate technical indicators using existing data
        live_data_processed = calculate_technical_indicators(live_data, existing_df)

        # Append new data and drop duplicates
        combined_df = pd.concat([existing_df, live_data_processed], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['code', 'date'], keep='last')
        combined_df = combined_df.sort_values(['code', 'date'])

        # Save updated CSV
        combined_df.to_csv(CSV_PATH, index=False)
        print(f"✅ Data for {curr_date.strftime('%Y-%m-%d')} added. Total records: {len(combined_df)}")

        # Update existing_df for next iteration
        existing_df = combined_df.copy()

        # Move to next day
        curr_date += timedelta(days=1)

    print("\nBackfill complete! Your CSV now includes all missed days.")

if __name__ == "__main__":
    main()
