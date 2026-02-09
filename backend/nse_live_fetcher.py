"""
NSE Kenya Live Data Fetcher - ROBUST VERSION
Multiple data sources with automatic fallback
Works reliably on GitHub Actions

Data sources (tried in order):
1. afx.kwayisi.org (primary)
2. live.mystocks.co.ke (fallback)

Usage:
    python nse_live_fetcher.py          # Test mode
    python nse_live_fetcher.py --update # Update database
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import argparse
import sys
import time
from io import StringIO


class NSELiveDataFetcher:
    """Fetch live NSE data with multiple source fallback"""
    
    SOURCES = [
        {
            'name': 'AFX',
            'url': 'https://afx.kwayisi.org/nse/',
            'parser': 'parse_afx'
        },
        {
            'name': 'MyStocks',
            'url': 'https://live.mystocks.co.ke/m/pricelist',
            'parser': 'parse_mystocks'
        }
    ]
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
    
    def parse_afx(self, html):
        """Parse AFX page"""
        soup = BeautifulSoup(html, 'html.parser')
        tables = pd.read_html(StringIO(html))
        
        # Find stocks table (largest table)
        stocks_table = None
        max_rows = 0
        
        for table in tables:
            if len(table) > max_rows and len(table) > 5:
                cols_str = ' '.join([str(c).lower() for c in table.columns])
                if any(kw in cols_str for kw in ['symbol', 'company', 'price', 'stock']):
                    stocks_table = table
                    max_rows = len(table)
        
        if stocks_table is None and len(tables) > 1:
            stocks_table = tables[1]
        
        if stocks_table is None:
            return None
        
        df = stocks_table.copy()
        df.columns = df.columns.str.strip()
        
        # Map columns
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
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Must have code and price
        if 'code' not in df.columns or 'day_price' not in df.columns:
            return None
        
        return df
    
    def parse_mystocks(self, html):
        """Parse MyStocks page"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find the price table
        table = soup.find('table', class_='pricelist') or soup.find('table', {'id': 'pricelist'})
        
        if not table:
            # Try any large table
            tables = soup.find_all('table')
            for t in tables:
                if len(t.find_all('tr')) > 10:
                    table = t
                    break
        
        if not table:
            return None
        
        # Extract data
        stocks_data = []
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 3:
                continue
            
            try:
                code = cols[0].text.strip()
                name = cols[1].text.strip() if len(cols) > 1 else code
                price = cols[2].text.strip().replace(',', '') if len(cols) > 2 else '0'
                change = cols[3].text.strip().replace(',', '') if len(cols) > 3 else '0'
                changepct = cols[4].text.strip().replace('%', '').replace(',', '') if len(cols) > 4 else '0'
                volume = cols[5].text.strip().replace(',', '') if len(cols) > 5 else '0'
                
                stocks_data.append({
                    'code': code,
                    'name': name,
                    'day_price': price,
                    'change': change,
                    'changepct': changepct,
                    'volume': volume
                })
            except:
                continue
        
        if not stocks_data:
            return None
        
        return pd.DataFrame(stocks_data)
    
    def fetch_all_stocks(self, max_retries=3):
        """
        Fetch NSE stocks trying multiple sources with fallback
        
        Returns:
            pandas DataFrame with stock data
        """
        for source in self.SOURCES:
            print(f"\nTrying {source['name']}...")
            
            for attempt in range(max_retries):
                try:
                    print(f"   Attempt {attempt+1}/{max_retries}...")
                    
                    response = requests.get(
                        source['url'], 
                        headers=self.headers, 
                        timeout=30,
                        allow_redirects=True
                    )
                    response.raise_for_status()
                    
                    parser_method = getattr(self, source['parser'])
                    df = parser_method(response.text)
                    
                    if df is None or len(df) == 0:
                        raise Exception("No data extracted")
                    
                    df = self.clean_data(df)
                    
                    if len(df) > 0:
                        print(f"✅ Successfully fetched {len(df)} stocks from {source['name']}")
                        return df
                    
                except requests.exceptions.Timeout:
                    print(f"   Timeout")
                except requests.exceptions.RequestException as e:
                    print(f"   Request error: {str(e)[:100]}")
                except Exception as e:
                    print(f"   Parsing error: {str(e)[:100]}")
                
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    print(f"   Waiting {wait}s...")
                    time.sleep(wait)
            
            print(f"   {source['name']} failed after {max_retries} attempts")
        
        print("\nAll data sources failed")
        return None
    
    def clean_data(self, df):
        """Clean and standardize the data"""
        df['date'] = datetime.now().strftime('%Y-%m-%d')
        
        if 'code' in df.columns:
            df = df[df['code'].notna()]
            df = df[~df['code'].astype(str).str.lower().str.contains('total|index|market', na=False)]
        
        if 'day_price' in df.columns:
            df['day_price'] = pd.to_numeric(df['day_price'], errors='coerce')
        
        if 'change' in df.columns:
            df['change'] = pd.to_numeric(df['change'], errors='coerce')
            if 'day_price' in df.columns:
                df['previous'] = df['day_price'] - df['change']
        
        if 'changepct' in df.columns:
            df['changepct'] = df['changepct'].astype(str).str.replace('%', '').str.strip()
            df['changepct'] = pd.to_numeric(df['changepct'], errors='coerce')
        
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        df = df[df['day_price'].notna()]
        df = df[df['day_price'] > 0]
        
        if 'day_high' not in df.columns:
            df['day_high'] = df['day_price']
        if 'day_low' not in df.columns:
            df['day_low'] = df['day_price']
        
        return df


def calculate_technical_indicators(df, historical_df=None):
    """Calculate technical indicators"""
    print("Calculating technical indicators...")
    
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
    
    print("Technical indicators calculated")
    return df


def update_database(new_data, db_path="data/processed/NSE_20_stocks_2013_2025_features_target.csv"):
    """Update database with new data"""
    try:
        print(f"Updating database at {db_path}...")
        
        existing_data = pd.read_csv(db_path, low_memory=False)
        existing_data['date'] = pd.to_datetime(existing_data['date'], errors='coerce')
        
        new_data_processed = calculate_technical_indicators(new_data, existing_data)
        new_data_processed['date'] = pd.to_datetime(new_data_processed['date'], errors='coerce')
        
        updated_data = pd.concat([existing_data, new_data_processed], ignore_index=True)
        updated_data = updated_data.drop_duplicates(subset=['code', 'date'], keep='last')
        updated_data = updated_data.sort_values(['code', 'date'])
        
        updated_data.to_csv(db_path, index=False)
        
        print(f"Database updated!")
        print(f"   Added {len(new_data_processed)} new records")
        print(f"   Total records: {len(updated_data)}")
        
        return True
        
    except Exception as e:
        print(f"Error updating database: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fetch live NSE data with automatic fallback')
    parser.add_argument('--update', action='store_true', help='Update database with live data')
    parser.add_argument('--db-path', default='data/processed/NSE_20_stocks_2013_2025_features_target.csv',
                       help='Path to database CSV file')
    
    args = parser.parse_args()
    
    fetcher = NSELiveDataFetcher()
    live_data = fetcher.fetch_all_stocks()
    
    if live_data is None or len(live_data) == 0:
        print("Failed to fetch data from all sources")
        sys.exit(1)
    
    print("\nSample of fetched data:")
    print(live_data.head(10))
    print(f"\nColumns: {live_data.columns.tolist()}")
    print(f"\nTotal stocks: {len(live_data)}")
    
    if args.update:
        success = update_database(live_data, args.db_path)
        if success:
            print("Live data successfully integrated!")
        else:
            print("Failed to update database")
            sys.exit(1)
    else:
        print("Run with --update to save this data to your database")


if __name__ == "__main__":
    main()
