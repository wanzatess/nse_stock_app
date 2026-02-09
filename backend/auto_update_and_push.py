"""
Final Auto-Update & Push Script for NSE_STOCK_APP
- Pulls latest changes from GitHub
- Fetches NSE data
- Commits & pushes CSV if updated
- Emoji-free and Windows-friendly
"""

import subprocess
import sys
from datetime import datetime
import os

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_DIR = r"C:\Users\HomePC\Documents\PROJECTS\NSE_STOCK_APP\backend"
FETCHER_PATH = os.path.join(PROJECT_DIR, "nse_live_fetcher.py")
# CSV is outside backend/ folder
CSV_PATH = os.path.join(os.path.dirname(PROJECT_DIR), "data", "processed", "NSE_20_stocks_2013_2025_features_target.csv")
BRANCH = "main"  # GitHub branch

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def run_command(cmd, cwd=PROJECT_DIR):
    """Run shell command and print output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("\n" + "="*70)
    print(f"NSE Auto-Update & Push Script")
    print(f"Timestamp: {datetime.now()}")
    print("="*70 + "\n")

    # Step 0: Ensure branch tracking
    print("Step 0: Ensure branch tracking...")
    run_command(f"git branch --set-upstream-to=origin/{BRANCH} {BRANCH}")

    # Step 1: Pull latest changes
    print("\nStep 1: Pulling latest changes from GitHub...")
    run_command(f"git pull origin {BRANCH}")

    # Step 2: Fetch NSE data
    print("\nStep 2: Fetching live NSE data...")
    if not os.path.exists(FETCHER_PATH):
        print(f"ERROR: Fetcher not found at {FETCHER_PATH}")
        sys.exit(1)
    
    if not run_command(f'python "{FETCHER_PATH}" --update'):
        print("\nData fetch failed! Check your internet connection or fetcher")
        sys.exit(1)

    # Step 3: Ensure CSV folder exists
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Step 4: Stage CSV changes
    print("\nStep 3: Staging CSV changes...")
    run_command(f'git add "{CSV_PATH}"')

    # Step 5: Check if there are changes
    result = subprocess.run("git diff --staged --quiet", shell=True, cwd=PROJECT_DIR)
    if result.returncode == 0:
        print("Info: No new data to commit (CSV unchanged)")
        return

    # Step 6: Commit changes
    print("\nStep 4: Committing changes...")
    commit_msg = f"Auto-update NSE data - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    run_command(f'git commit -m "{commit_msg}"')

    # Step 7: Push to GitHub
    print("\nStep 5: Pushing to GitHub...")
    if run_command(f"git push origin {BRANCH}"):
        print("\nSUCCESS! Data pushed to GitHub")
        print("Render will auto-deploy in 2-3 minutes")
    else:
        print("\nPush failed - check git credentials or network")

    print("\n" + "="*70 + "\n")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()
