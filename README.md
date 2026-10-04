# SignalGuard

A Python dashboard for evaluating a stock-direction model,
simulating a trading strategy, and investigating its failures.

## Features

- CSV validation and historical price charts
- Moving averages and model-feature preparation
- Random Forest classification with chronological testing
- Five-period walk-forward evaluation
- Confusion matrix and baseline comparison
- Next-open trading simulation with transaction costs
- Drawdown analysis and a failure explorer
- Downloadable results

## Setup on Windows

## Run locally

Open a terminal in the downloaded or cloned project folder.

Create a virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download example data:

```powershell
.\.venv\Scripts\python.exe download_data.py
```

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload `AAPL_2021_2025.csv` in the dashboard.

## Input format

One stock per CSV, with Date, Open, and Close columns.
Dates must use YYYY-MM-DD.
Opening and closing prices must be positive and use the same
adjustment basis.

The supplied downloader retrieves adjusted Apple prices for
2021–2025.

## Method

The model predicts whether the next closing price will rise.
Features use information available at the current close.

Training uses older records; testing uses later records.
A one-record gap separates training and testing.

The simulated strategy holds stock after an up prediction and
cash otherwise. Position changes occur at the next open.
Returns are measured from open to open.

Assumptions:

- $10,000 starting capital
- 0.1% cost per purchase or sale
- Fractional holdings
- Zero interest on cash
- No taxes
- Final holdings liquidated

## Recorded example results

For the tested Apple dataset:

- Model accuracy: 48.4%
- Baseline accuracy: 52.4%
- Strategy return after costs: -3.97%
- Buy-and-hold return after costs: +9.97%
- Strategy maximum drawdown: -29.53%
- Buy-and-hold maximum drawdown: -30.92%
- Strategy orders: 94

The model did not demonstrate an advantage in this evaluation.
Results may change with data revisions or dependency versions.

## Limitations

This is an educational research prototype.

The model predicts close-to-close direction, while the strategy
earns open-to-open returns.

The test period has already been inspected and is not a fresh,
independent test for subsequent model changes.

Drawdowns use sampled portfolio values and exclude intraday falls.
Adjusted prices are a simplified basis for simulation.

## Validation performed

- Main dashboard workflows manually checked
- Exported backtest independently reconciled across 249 intervals
- Costs, balances, returns, drawdowns, and date continuity checked

These checks do not independently verify the market-data source.
