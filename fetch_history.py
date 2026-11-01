"""Download real day-ahead spot price history for the configured bidding zone (default DE-LU) to PRICE_HISTORY_PATH.

Usage: python fetch_history.py
Source: Energy-Charts (Fraunhofer ISE), data Bundesnetzagentur | SMARD.de, CC BY 4.0. No API key needed.
If the download fails, nothing is written: the app never trains on invented prices.
The backend also refreshes this file automatically (data/price_history.py:ensure_fresh).
"""
import sys

from config.settings import get_settings
from data.price_history import refresh_history

if __name__ == "__main__":
    s = get_settings()
    try:
        n = refresh_history(s)
    except Exception as exc:
        print(f"Failed to fetch price history: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Saved {n} real {s.BIDDING_ZONE} day-ahead prices to {s.PRICE_HISTORY_PATH}")
