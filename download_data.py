from pathlib import Path

import yfinance as yf

print("Downloading Apple daily prices...")

try:
    prices = yf.download(
        "AAPL",
        start="2021-01-01",
        end="2026-01-01",
        interval="1d",
        auto_adjust=True,
        multi_level_index=False,
        progress=False,
    )
except Exception as error:
    raise SystemExit(f"Download failed: {error}")

if prices is None or prices.empty:
    raise SystemExit(
        "No data received. Check your internet connection "
        "and try again later."
    )

# Keep the two columns our dashboard expects.
data = prices[["Open", "Close"]].copy()
data.index.name = "Date"
data = data.reset_index()

output_path = Path(__file__).resolve().parent / "AAPL_2021_2025.csv"

data.to_csv(
    output_path,
    index=False,
    date_format="%Y-%m-%d",
)

print(f"Saved {len(data)} records.")
print(f"File: {output_path}")
print(data.head().to_string(index=False))