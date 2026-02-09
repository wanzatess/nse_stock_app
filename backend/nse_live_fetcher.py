"""
NSE Kenya Live Data Fetcher - Investing.com Version
Scrapes real-time stock data from investing.com (more reliable for GitHub Actions)

Usage:
    python nse_live_fetcher_investing.py          # Test mode
    python nse_live_fetcher_investing.py --update # Update database
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import argparse
import sys
import time
import json

class NSELiveDataFetcher:
    """Fetch live NSE data from Investing.com"""
    
    BASE_URL = "https://www.investing.com/equities/kenya"
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
    
    def fetch_all_stocks(self, max_retries=3):
        """
        Fetch all NSE stocks from Investing.com
        
        Returns:
            pandas DataFrame with columns: code, name, day_price, change, changepct, volume, etc.
        """
        for attempt in range(max_retries):
            try:
                print(f"📡 Fetching live NSE data from Investing.com (attempt {attempt+1}/{max_retries})...")
                
                response = requests.get(self.BASE_URL, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the stocks table
                # Investing.com uses a table with class containing 'genTbl' or similar
                table = soup.find('table', {'id': 'cr1'}) or soup.find('table', class_=lambda x: x and 'genTbl' in x)
                
                if not table:
                    print("⚠️  Could not find stocks table, trying alternative method...")
                    # Try to find any table with stock data
                    tables = soup.find_all('table')
                    for t in tables:
                        if len(t.find_all('tr')) > 10:  # Should have many rows
                            table = t
                            break
                
                if not table:
                    raise Exception("Could not find stocks table on page")
                
                # Extract data
                stocks_data = []
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 5:
                        continue
                    
                    try:
                        # Extract stock name and code
                        name_cell = cols[0]
                        name_link = name_cell.find('a')
                        if name_link:
                            full_name = name_link.text.strip()
                            # Try to extract code from link or name
                            link_href = name_link.get('href', '')
                            code = link_href.split('/')[-1].upper() if link_href else full_name.split()[0]
                        else:
                            full_name = name_cell.text.strip()
                            code = full_name.split()[0]
                        
                        # Extract price and changes
                        price = cols[1].text.strip().replace(',', '')
                        change = cols[2].text.strip().replace(',', '')
                        changepct = cols[3].text.strip().replace('%', '').replace(',', '')
                        
                        # Volume (if available)
                        volume = cols[4].text.strip().replace(',', '') if len(cols) > 4 else '0'
                        
                        stocks_data.append({
                            'code': code,
                            'name': full_name,
                            'day_price': price,
                            'change': change,
                            'changepct': changepct,
                            'volume': volume
                        })
                    
                    except Exception as e:
                        print(f"  ⚠️  Error parsing row: {e}")
                        continue
                
                if not stocks_data:
                    raise Exception("No stock data extracted from table")
                
                # Create DataFrame
                df = pd.DataFrame(stocks_data)
                
                # Clean and convert data types
                df['day_price'] = pd.to_numeric(df['day_price'], errors='coerce')
                df['change'] = pd.to_numeric(df['change'], errors='coerce')
                df['changepct'] = pd.to_numeric(df['changepct'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                
                # Calculate previous price
                df['previous'] = df['day_price'] - df['change']
                
                # Add date
                df['date'] = datetime.now().strftime('%Y-%m-%d')
                
                # Add day high/low (use current price as estimate)
                df['day_high'] = df['day_price']
                df['day_low'] = df['day_price']
                
                # Drop rows with invalid data
                df = df[df['day_price'].notna()]
                df = df[df['day_price'] > 0]
                
                print(f"✅ Fetched data for {len(df)} stocks")
                print(f"Columns: {df.columns.tolist()}")
                
                return df
                
            except Exception as e:
                print(f"❌ Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("❌ All retry attempts failed")
                    import traceback
                    traceback.print_exc()
                    return None
        
        return None


def calculate_technical_indicators(df, historical_df=None):
    """Calculate technical indicators (same as before)"""
    print("📊 Calculating technical indicators...")
    
    df = df.copy()
    
    if historical_df is not None and not historical_df.empty:
        historical_df = historical_df.copy()
        historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        recent_history = historical_df.sort_values('date').tail(365 * len(df['code'].unique()))
        combined = pd.concat([recent_history, df], ignore_index=True)
    else:
        combined = df.copy()
    
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined = combined.sort_values(['code', 'date'])
    
    for code in df['code'].unique():
        stock_data = combined[combined['code'] == code].copy()
        
        if len(stock_data) < 2:
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
        
        stock_data['ma_5'] = stock_data['day_price'].rolling(window=5, min_periods=1).mean()
        stock_data['ma_10'] = stock_data['day_price'].rolling(window=10, min_periods=1).mean()
        stock_data['daily_return'] = stock_data['day_price'].pct_change() * 100
        stock_data['daily_volatility'] = stock_data['day_price'].pct_change().rolling(window=30, min_periods=1).std() * 100
        stock_data['12m_high'] = stock_data['day_price'].rolling(window=252, min_periods=1).max()
        stock_data['12m_low'] = stock_data['day_price'].rolling(window=252, min_periods=1).min()
        stock_data['pct_from_12m_high'] = ((stock_data['day_price'] - stock_data['12m_high']) / stock_data['12m_high']) * 100
        stock_data['pct_from_12m_low'] = ((stock_data['day_price'] - stock_data['12m_low']) / stock_data['12m_low']) * 100
        
        latest_row = stock_data.iloc[-1]
        mask = df['code'] == code
        
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


def update_database(new_data, db_path="data/processed/NSE_20_stocks_2013_2025_features_target.csv"):
    """Update database with new data"""
    try:
        print(f"💾 Updating database at {db_path}...")
        
        existing_data = pd.read_csv(db_path, low_memory=False)
        existing_data['date'] = pd.to_datetime(existing_data['date'], errors='coerce')
        
        new_data_processed = calculate_technical_indicators(new_data, existing_data)
        new_data_processed['date'] = pd.to_datetime(new_data_processed['date'], errors='coerce')
        
        updated_data = pd.concat([existing_data, new_data_processed], ignore_index=True)
        updated_data = updated_data.drop_duplicates(subset=['code', 'date'], keep='last')
        updated_data = updated_data.sort_values(['code', 'date'])
        
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
    parser = argparse.ArgumentParser(description='Fetch live NSE data from Investing.com')
    parser.add_argument('--update', action='store_true', help='Update database with live data')
    parser.add_argument('--db-path', default='data/processed/NSE_20_stocks_2013_2025_features_target.csv',
                       help='Path to database CSV file')
    
    args = parser.parse_args()
    
    fetcher = NSELiveDataFetcher()
    live_data = fetcher.fetch_all_stocks()
    
    if live_data is None or len(live_data) == 0:
        print("❌ Failed to fetch data or no stocks found")
        sys.exit(1)
    
    print("\n📋 Sample of fetched data:")
    print(live_data.head(10))
    print(f"\n📊 Columns: {live_data.columns.tolist()}")
    print(f"\n📈 Total stocks: {len(live_data)}")
    
    if args.update:
        success = update_database(live_data, args.db_path)
        if success:
            print("\n🎉 Live data successfully integrated!")
        else:
            print("\n❌ Failed to update database")
            sys.exit(1)
    else:
        print("\n💡 Tip: Run with --update to save this data to your database")


if __name__ == "__main__":
    main()