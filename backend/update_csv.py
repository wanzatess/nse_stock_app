import subprocess
import os

def main():
    # Path to your fetcher
    fetcher_path = os.path.join(os.path.dirname(__file__), "nse_live_fetcher.py")

    # Path to database CSV
    db_path = os.path.join(os.path.dirname(__file__), "data/processed/NSE_20_stocks_2013_2025_features_target.csv")

    # Run fetcher with --update
    cmd = f"python \"{fetcher_path}\" --update --db-path \"{db_path}\""
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
