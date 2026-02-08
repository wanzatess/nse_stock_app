"""
NSE Kenya Live Data Fetcher - FIXED VERSION
Scrapes real-time stock data from afx.kwayisi.org

Usage:
    python nse_live_fetcher_fixed.py          # Test mode - shows sample data
    python nse_live_fetcher_fixed.py --update # Update your database with live data
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import argparse
import sys
from io import StringIO

class NSELiveDataFetcher:
    """Fetch live NSE data from AFX"""
    
    BASE_URL = "https://afx.kwayisi.org/nse/"
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    
    def fetch_all_stocks(self):
        """
        Fetch all NSE stocks from the main AFX page
        
        Returns:
            pandas DataFrame with columns: code, name, day_price, change, changepct, volume, etc.
        """
        try:
            print("📡 Fetching live NSE data from AFX...")
            response = requests.get(self.BASE_URL, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find ALL tables on the page
            # Fix the FutureWarning by using StringIO
            tables = pd.read_html(StringIO(response.text))
            
            print(f"Found {len(tables)} tables on the page")
            
            # The stocks table is usually the largest one (not the first one which is the index summary)
            # Find the table with the most rows
            stocks_table = None
            max_rows = 0
            
            for i, table in enumerate(tables):
                print(f"Table {i}: {len(table)} rows, columns: {table.columns.tolist()}")
                if len(table) > max_rows and len(table) > 5:  # Must have at least 5 stocks
                    # Check if it looks like a stocks table
                    # It should have columns like Symbol, Price, Change, etc.
                    cols_str = ' '.join([str(c).lower() for c in table.columns])
                    if any(keyword in cols_str for keyword in ['symbol', 'company', 'price', 'stock']):
                        stocks_table = table
                        max_rows = len(table)
                        print(f"  ✓ This looks like the stocks table!")
            
            if stocks_table is None:
                print("❌ Could not find stocks table")
                # Try to use the second table if it exists
                if len(tables) > 1:
                    print("Trying second table as fallback...")
                    stocks_table = tables[1]
                else:
                    return None
            
            df = stocks_table.copy()
            
            # Clean up column names (remove extra spaces, etc.)
            df.columns = df.columns.str.strip()
            
            print(f"\n📊 Raw columns: {df.columns.tolist()}")
            print(f"First few rows:\n{df.head()}")
            
            # Standardize column names to match your database
            # AFX typically has columns like: Symbol/Code, Company/Name, Price, Change, %, Volume, etc.
            column_mapping = {}
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'symbol' in col_lower or 'code' in col_lower or 'ticker' in col_lower:
                    column_mapping[col] = 'code'
                elif 'company' in col_lower or 'name' in col_lower:
                    column_mapping[col] = 'name'
                elif 'price' in col_lower or 'close' in col_lower:
                    column_mapping[col] = 'day_price'
                elif 'change' in col_lower and '%' not in col_lower:
                    column_mapping[col] = 'change'
                elif '%' in col_lower or 'percent' in col_lower:
                    column_mapping[col] = 'changepct'
                elif 'volume' in col_lower or 'vol' in col_lower:
                    column_mapping[col] = 'volume'
                elif 'high' in col_lower:
                    column_mapping[col] = 'day_high'
                elif 'low' in col_lower:
                    column_mapping[col] = 'day_low'
            
            print(f"\n🔄 Column mapping: {column_mapping}")
            
            # Rename columns
            df.rename(columns=column_mapping, inplace=True)
            
            # Ensure we have the essential columns
            required = ['code', 'day_price']
            missing = [col for col in required if col not in df.columns]
            if missing:
                print(f"❌ Missing required columns: {missing}")
                print(f"Available columns: {df.columns.tolist()}")
                return None
            
            # Add date
            df['date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Clean the data
            # Remove any summary rows (often marked with "Total" or contains non-numeric data)
            if 'code' in df.columns:
                df = df[df['code'].notna()]
                df = df[~df['code'].astype(str).str.lower().str.contains('total|index|market', na=False)]
            
            # Convert price to numeric
            if 'day_price' in df.columns:
                df['day_price'] = pd.to_numeric(df['day_price'], errors='coerce')
            
            # Calculate previous price if we have change
            if 'change' in df.columns and 'day_price' in df.columns:
                df['change'] = pd.to_numeric(df['change'], errors='coerce')
                df['previous'] = df['day_price'] - df['change']
            
            # Clean percentage values (remove % sign if present)
            if 'changepct' in df.columns:
                df['changepct'] = df['changepct'].astype(str).str.replace('%', '').str.strip()
                df['changepct'] = pd.to_numeric(df['changepct'], errors='coerce')
            
            # Clean volume
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Drop rows with invalid price data
            df = df[df['day_price'].notna()]
            df = df[df['day_price'] > 0]
            
            # If we don't have day_high and day_low, use the current price
            if 'day_high' not in df.columns:
                df['day_high'] = df['day_price']
            if 'day_low' not in df.columns:
                df['day_low'] = df['day_price']
            
            print(f"\n✅ Fetched data for {len(df)} stocks")
            print(f"Final columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return None


def calculate_technical_indicators(df, historical_df=None):
    """
    Calculate technical indicators (MA, volatility, etc.) for the new data
    
    Args:
        df: New data DataFrame
        historical_df: Historical data for calculations (optional)
        
    Returns:
        DataFrame with calculated features
    """
    print("📊 Calculating technical indicators...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # If we have historical data, combine for better calculations
    if historical_df is not None and not historical_df.empty:
        # Ensure date column is datetime
        historical_df = historical_df.copy()
        historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Get last 365 days of historical data
        recent_history = historical_df.sort_values('date').tail(365 * len(df['code'].unique()))
        combined = pd.concat([recent_history, df], ignore_index=True)
    else:
        combined = df.copy()
    
    # Ensure date is datetime
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    
    # Sort by code and date
    combined = combined.sort_values(['code', 'date'])
    
    # Calculate for each stock
    for code in df['code'].unique():
        stock_data = combined[combined['code'] == code].copy()
        
        if len(stock_data) < 2:
            print(f"  ⚠️  Not enough data for {code}, using defaults")
            # Set default values
            mask = df['code'] == code
            df.loc[mask, 'ma_5'] = df.loc[mask, 'day_price']
            df.loc[mask, 'ma_10'] = df.loc[mask, 'day_price']
            df.loc[mask, 'daily_return'] = 0.0
            df.loc[mask, 'daily_volatility'] = 0.0
            df.loc[mask, '12m_high'] = df.loc[mask, 'day_price']
            df.loc[mask, '12m_low'] = df.loc[mask, 'day_price']
            df.loc[mask, 'pct_from_12m_high'] = 0.0
            df.loc[mask, 'pct_from_12m_low'] = 0.0
            continue
        
        # Calculate moving averages
        stock_data['ma_5'] = stock_data['day_price'].rolling(window=5, min_periods=1).mean()
        stock_data['ma_10'] = stock_data['day_price'].rolling(window=10, min_periods=1).mean()
        
        # Calculate daily return
        stock_data['daily_return'] = stock_data['day_price'].pct_change() * 100
        
        # Calculate volatility (30-day std of returns)
        stock_data['daily_volatility'] = stock_data['day_price'].pct_change().rolling(window=30, min_periods=1).std() * 100
        
        # 12-month high/low
        stock_data['12m_high'] = stock_data['day_price'].rolling(window=252, min_periods=1).max()
        stock_data['12m_low'] = stock_data['day_price'].rolling(window=252, min_periods=1).min()
        
        # Distance from 12m high/low
        stock_data['pct_from_12m_high'] = ((stock_data['day_price'] - stock_data['12m_high']) / stock_data['12m_high']) * 100
        stock_data['pct_from_12m_low'] = ((stock_data['day_price'] - stock_data['12m_low']) / stock_data['12m_low']) * 100
        
        # Get the latest row for this stock
        latest_row = stock_data.iloc[-1]
        mask = df['code'] == code
        
        # Update the main dataframe with calculated values (only for new rows)
        df.loc[mask, 'ma_5'] = latest_row['ma_5']
        df.loc[mask, 'ma_10'] = latest_row['ma_10']
        df.loc[mask, 'daily_return'] = latest_row['daily_return'] if pd.notna(latest_row['daily_return']) else 0.0
        df.loc[mask, 'daily_volatility'] = latest_row['daily_volatility'] if pd.notna(latest_row['daily_volatility']) else 0.0
        df.loc[mask, '12m_high'] = latest_row['12m_high']
        df.loc[mask, '12m_low'] = latest_row['12m_low']
        df.loc[mask, 'pct_from_12m_high'] = latest_row['pct_from_12m_high'] if pd.notna(latest_row['pct_from_12m_high']) else 0.0
        df.loc[mask, 'pct_from_12m_low'] = latest_row['pct_from_12m_low'] if pd.notna(latest_row['pct_from_12m_low']) else 0.0
    
    print("✅ Technical indicators calculated")
    return df


def update_database(new_data, db_path="../data/processed/NSE_20_stocks_2013_2025_features_target.csv"):
    """
    Update the main database with new data
    
    Args:
        new_data: DataFrame with new stock data
        db_path: Path to the database CSV file
    """
    try:
        print(f"💾 Updating database at {db_path}...")
        
        # Load existing database
        existing_data = pd.read_csv(db_path, low_memory=False)
        existing_data['date'] = pd.to_datetime(existing_data['date'], errors='coerce')
        
        # Calculate technical indicators for new data
        new_data_processed = calculate_technical_indicators(new_data, existing_data)
        
        # Ensure date column is datetime in new data
        new_data_processed['date'] = pd.to_datetime(new_data_processed['date'], errors='coerce')
        
        # Append new data
        updated_data = pd.concat([existing_data, new_data_processed], ignore_index=True)
        
        # Remove duplicates (keep latest)
        updated_data = updated_data.drop_duplicates(subset=['code', 'date'], keep='last')
        
        # Sort by date
        updated_data = updated_data.sort_values(['code', 'date'])
        
        # Save
        updated_data.to_csv(db_path, index=False)
        
        print(f"✅ Database updated!")
        print(f"   Added {len(new_data_processed)} new records")
        print(f"   Total records: {len(updated_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating database: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fetch live NSE data from AFX')
    parser.add_argument('--update', action='store_true', help='Update database with live data')
    parser.add_argument('--db-path', default='../data/processed/NSE_20_stocks_2013_2025_features_target.csv',
                       help='Path to database CSV file')
    
    args = parser.parse_args()
    
    # Create fetcher
    fetcher = NSELiveDataFetcher()
    
    # Fetch data
    live_data = fetcher.fetch_all_stocks()
    
    if live_data is None or len(live_data) == 0:
        print("❌ Failed to fetch data or no stocks found")
        sys.exit(1)
    
    # Display sample
    print("\n📋 Sample of fetched data:")
    print(live_data.head(10))
    print(f"\n📊 Columns: {live_data.columns.tolist()}")
    print(f"\n📈 Total stocks: {len(live_data)}")
    
    # Update database if requested
    if args.update:
        success = update_database(live_data, args.db_path)
        if success:
            print("\n🎉 Live data successfully integrated!")
        else:
            print("\n❌ Failed to update database")
            sys.exit(1)
    else:
        print("\n💡 Tip: Run with --update to save this data to your database")
        print("   Example: python nse_live_fetcher_fixed.py --update")


if __name__ == "__main__":
    main()