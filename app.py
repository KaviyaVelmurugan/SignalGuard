import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
st.set_page_config(
    page_title="SignalGuard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 SignalGuard")
st.write("Upload historical stock prices to explore your data.")

st.sidebar.title("Project progress")
st.sidebar.success("Dashboard created")
st.sidebar.info("Current step: Phase 5 — Risk and failure analysis")

uploaded_file = st.file_uploader(
    "Choose a CSV file containing Date, Open, and Close columns",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

# Read the uploaded file.
try:
    df = pd.read_csv(uploaded_file)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
    st.error("Could not read this file. Please upload a valid UTF-8 CSV.")
    st.stop()

# Remove accidental spaces around column names.
df.columns = df.columns.str.strip()
required_columns = {"Date", "Open", "Close"}
if not required_columns.issubset(df.columns):
    st.error("Your CSV must contain columns named Date, Open, and Close.")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

df = df[["Date", "Open", "Close"]].copy()

# Convert text into dates and numbers.
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%Y-%m-%d",
    errors="coerce",
)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
# Reject invalid records instead of silently removing them.
invalid_rows = (
    df["Date"].isna()
    | df["Close"].isna()
    | df["Open"].isna()
    | (df["Close"] <= 0)
    | (df["Open"] <= 0)
    | df["Close"].isin([float("inf"), float("-inf")])
    | df["Open"].isin([float("inf"), float("-inf")])
)

if invalid_rows.any():
    st.error(
        f"Found {int(invalid_rows.sum())} invalid rows. "
        "Use dates like 2025-01-02 and positive, finite opening and closing prices."
    )
    st.dataframe(df.loc[invalid_rows], hide_index=True)
    st.stop()

if df["Date"].duplicated().any():
    st.error("Duplicate dates found. Keep one closing price per date.")
    st.stop()

df = df.sort_values("Date").reset_index(drop=True)

if len(df) < 2:
    st.error("Please upload at least two rows of prices.")
    st.stop()

st.success(f"Loaded {len(df)} price records.")

# Summarize the uploaded period.
first_close = df["Close"].iloc[0]
latest_close = df["Close"].iloc[-1]
period_change = (latest_close / first_close - 1) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Price records", len(df))
col2.metric("Latest closing price", f"{latest_close:,.2f}")
col3.metric("Price change over uploaded period", f"{period_change:+.2f}%")

# Calculate averages using only the current and earlier records.
df["SMA_3"] = df["Close"].rolling(window=3).mean()
df["SMA_5"] = df["Close"].rolling(window=5).mean()

st.subheader("Price and moving averages")

st.line_chart(
    df.set_index("Date")[["Close", "SMA_3", "SMA_5"]]
)

st.caption(
    "SMA_3 averages the latest 3 trading records. "
    "SMA_5 averages the latest 5 trading records."
)

if len(df) >= 5:
    latest = df.iloc[-1]

    col1, col2 = st.columns(2)
    col1.metric("3-record average", f"{latest['SMA_3']:.2f}")
    col2.metric("5-record average", f"{latest['SMA_5']:.2f}")

    if latest["SMA_3"] > latest["SMA_5"]:
        st.info(
            "The shorter average is above the longer average, "
            "indicating stronger recent prices."
        )
    elif latest["SMA_3"] < latest["SMA_5"]:
        st.info(
            "The shorter average is below the longer average, "
            "indicating weaker recent prices."
        )
    else:
        st.info("The two moving averages are currently equal.")
else:
    st.info("Upload at least 5 records to compare both averages.")

st.subheader("Uploaded data")
st.dataframe(df, hide_index=True, use_container_width=True)

st.caption(
    "This file uses adjusted historical prices. "
    "Displayed price change is not a simulated trading return "
    "and does not deduct trading costs."
)
st.divider()
st.subheader("Prepare data for machine learning")

# Features: information available at the current close.
df["Return_1"] = df["Close"].pct_change(fill_method=None)
df["Return_previous"] = df["Return_1"].shift(1)
df["Distance_SMA_3"] = df["Close"] / df["SMA_3"] - 1
df["Distance_SMA_5"] = df["Close"] / df["SMA_5"] - 1

feature_columns = [
    "Return_1",
    "Return_previous",
    "Distance_SMA_3",
    "Distance_SMA_5",
]

# Target: did the next closing price rise?
# Future prices are used ONLY for labels, never as features.
next_close = df["Close"].shift(-1)

df["Target"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
known_outcome = next_close.notna()

df.loc[known_outcome, "Target"] = (
    next_close.loc[known_outcome]
    > df.loc[known_outcome, "Close"]
).astype(int)

# Training examples require complete features and a known outcome.
training_data = df.dropna(
    subset=feature_columns + ["Target"]
).copy()

st.write(
    "Target 1 means the next closing price rose. "
    "Target 0 means it stayed the same or fell."
)

st.dataframe(
    training_data[["Date", "Close"] + feature_columns + ["Target"]],
    hide_index=True,
)

st.metric("Complete labeled examples", len(training_data))

# Keep the latest record separate for a future prediction.
latest_features = df.tail(1)[feature_columns]

if latest_features.notna().all().all():
    st.caption(
        "The latest record is reserved for prediction because "
        "its next closing price is not yet available."
    )

if len(training_data) < 200:
    st.info(
        "This is enough to inspect the preparation step. "
        "Before model training, we’ll load a longer real price history. "
        "We’ll use 200 labeled examples as a tutorial minimum, "
        "not a guarantee of reliable results."
    )

st.divider()
st.subheader("Data preparation summary")

if training_data.empty:
    st.warning(
        "No complete training examples yet. "
        "Upload at least 6 valid price records."
    )
else:
    up_count = int((training_data["Target"] == 1).sum())
    not_up_count = int((training_data["Target"] == 0).sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Usable examples", len(training_data))
    col2.metric("Next price rose", up_count)
    col3.metric("Next price stayed equal or fell", not_up_count)

    st.write(
        "Prepared date range:",
        training_data["Date"].min().strftime("%Y-%m-%d"),
        "to",
        training_data["Date"].max().strftime("%Y-%m-%d"),
    )

    # Export only the date, price, model inputs, and known target.
    prepared_data = training_data[
        ["Date", "Close"] + feature_columns + ["Target"]
    ].copy()

    st.download_button(
        label="Download prepared data",
        data=prepared_data.to_csv(
            index=False,
            date_format="%Y-%m-%d",
        ).encode("utf-8"),
        file_name="signalguard_prepared_data.csv",
        mime="text/csv",
    )

    st.success(
        "Data preparation finished for this file: "
        "features calculated and unknown outcomes excluded."
    )

    if up_count == 0 or not_up_count == 0:
        st.warning(
            "This file contains only one outcome category. "
            "Use a longer history with both rising and non-rising outcomes "
            "before training."
        )

    if len(training_data) < 200:
        st.info(
            "This small dataset is for checking the workflow. "
            "Next, we’ll obtain a longer real history for model training."
        )

st.divider()
st.subheader("Train and evaluate the first model")

st.write(
    "Train on older records and test on newer records. "
    "The test outcomes are not used to train the model."
)

if len(training_data) < 200:
    st.info("Upload the real historical dataset before training.")

elif st.button("Train and evaluate"):
    model_data = training_data.sort_values("Date").reset_index(drop=True)

    split_index = int(len(model_data) * 0.8)

    # Skip one example at the boundary because each target uses
    # the following record's closing price.
    train = model_data.iloc[:split_index - 1].copy()
    test = model_data.iloc[split_index:].copy()

    X_train = train[feature_columns]
    y_train = train["Target"].astype(int)

    X_test = test[feature_columns]
    y_test = test["Target"].astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        st.warning(
            "Both periods need rising and non-rising outcomes "
            "for this evaluation. Try a longer dataset."
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

        with st.spinner("Training and evaluating..."):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        # Baseline: always predict the most common training outcome.
        baseline_class = int(y_train.mode().iloc[0])
        baseline_predictions = [baseline_class] * len(y_test)

        model_accuracy = accuracy_score(y_test, predictions)
        baseline_accuracy = accuracy_score(
            y_test, baseline_predictions
        )
        balanced_accuracy = balanced_accuracy_score(
            y_test, predictions
        )

        st.write(
            f"Training: {train['Date'].iloc[0]:%Y-%m-%d} to "
            f"{train['Date'].iloc[-1]:%Y-%m-%d} "
            f"({len(train)} examples)"
        )

        st.write(
            f"Testing: {test['Date'].iloc[0]:%Y-%m-%d} to "
            f"{test['Date'].iloc[-1]:%Y-%m-%d} "
            f"({len(test)} examples)"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Model accuracy", f"{model_accuracy:.1%}")
        col2.metric("Baseline accuracy", f"{baseline_accuracy:.1%}")
        col3.metric("Balanced accuracy", f"{balanced_accuracy:.1%}")

        st.caption(
            "Balanced accuracy gives equal weight to detecting "
            "rising and non-rising outcomes."
        )

        results = test[["Date", "Close"]].copy()
        results["Actual outcome"] = y_test.to_numpy()
        results["Predicted outcome"] = predictions
        results["Correct"] = (
            results["Actual outcome"] == results["Predicted outcome"]
        )
        st.subheader("Where the model gets predictions wrong")

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        matrix_table = pd.DataFrame(
            matrix,
            index=["Actually not up (0)", "Actually up (1)"],
            columns=["Predicted not up (0)", "Predicted up (1)"],
        )

        st.dataframe(matrix_table)

        st.caption(
            "Rows show what actually happened. "
            "Columns show what the model predicted. "
            "The top-left and bottom-right cells are correct predictions."
        )

        prediction_counts = pd.DataFrame({
            "Outcome": ["Not up (0)", "Up (1)"],
            "Actual count": [
                int((y_test == 0).sum()),
                int((y_test == 1).sum()),
            ],
            "Predicted count": [
                int((predictions == 0).sum()),
                int((predictions == 1).sum()),
            ],
        })

        st.subheader("Does the model favor one prediction?")
        st.dataframe(prediction_counts, hide_index=True)
        st.subheader("Most recent test predictions")
        st.dataframe(results.tail(20), hide_index=True)

        st.caption(
            "1 = next close rose; 0 = next close stayed equal or fell. "
            "These are predictions for historical test records."
        )

        st.info(
            "Prediction accuracy does not measure trading profit. "
            "We’ll evaluate simulated trades and costs in the next phase."
        )

st.divider()
st.subheader("Evaluate across multiple time periods")

st.write(
    "Test the same model settings across five chronological periods. "
    "Each model learns only from records before its test period."
)

if len(training_data) < 200:
    st.info("Upload the real historical dataset to continue.")

elif st.button("Run five-period evaluation"):
    evaluation_data = (
        training_data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Reserve the final 20% already used in our first evaluation.
    # Run this diagnostic on the earlier portion only.
    development_end = int(len(evaluation_data) * 0.8)

    # Exclude the boundary example whose label reaches into
    # the reserved period.
    development_data = evaluation_data.iloc[
        :development_end - 1
    ].copy()

    # A one-record gap separates training and testing because
    # our target uses the following record's closing price.
    splitter = TimeSeriesSplit(n_splits=5, gap=1)

    rows = []

    with st.spinner("Evaluating five time periods..."):
        for period, (train_indices, test_indices) in enumerate(
            splitter.split(development_data),
            start=1,
        ):
            train_part = development_data.iloc[train_indices]
            test_part = development_data.iloc[test_indices]

            X_train = train_part[feature_columns]
            y_train = train_part["Target"].astype(int)

            X_test = test_part[feature_columns]
            y_test = test_part["Target"].astype(int)

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                st.warning(
                    f"Period {period} skipped: both outcome "
                    "categories are needed."
                )
                continue

            period_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )

            period_model.fit(X_train, y_train)
            predictions = period_model.predict(X_test)

            baseline_class = int(y_train.mode().iloc[0])
            baseline_predictions = [baseline_class] * len(y_test)

            model_score = accuracy_score(y_test, predictions)
            baseline_score = accuracy_score(
                y_test, baseline_predictions
            )

            rows.append({
                "Period": period,
                "Test start": test_part["Date"].iloc[0].strftime(
                    "%Y-%m-%d"
                ),
                "Test end": test_part["Date"].iloc[-1].strftime(
                    "%Y-%m-%d"
                ),
                "Test examples": len(test_part),
                "Model accuracy (%)": model_score * 100,
                "Baseline accuracy (%)": baseline_score * 100,
                "Balanced accuracy (%)": (
                    balanced_accuracy_score(y_test, predictions) * 100
                ),
                "Difference (percentage points)": (
                    model_score - baseline_score
                ) * 100,
            })

    if rows:
        scores = pd.DataFrame(rows)

        st.dataframe(scores.round(2), hide_index=True)

        wins = int(
            (scores["Difference (percentage points)"] > 0).sum()
        )

        st.write(
            f"The model beat its baseline in "
            f"{wins} of {len(scores)} evaluated periods."
        )

        st.bar_chart(
            scores.set_index("Period")[
                ["Model accuracy (%)", "Baseline accuracy (%)"]
            ]
        )

        st.caption(
            "These are development-period diagnostics. "
            "The final 20% from the first evaluation is excluded here. "
            "That final period has already been inspected, so it is "
            "not a fresh independent test for future changes."
        )

st.divider()
st.subheader("Backtest timing preview")

timing = df[["Date", "Close"]].copy()
timing = timing.rename(columns={
    "Date": "Prediction date",
    "Close": "Close known at prediction",
})

timing["Next trading date"] = df["Date"].shift(-1)
timing["Next opening price"] = df["Open"].shift(-1)

st.dataframe(timing.tail(6), hide_index=True)

st.caption(
    "A prediction made after one day's close can first be acted on "
    "at the following trading day's open in our simulation. "
    "The final row has no next opening price in this file."
)
st.divider()
st.subheader("Trading simulation")

st.write(
    "Start with $10,000. Trade at the next opening price, "
    "with an assumed 0.1% cost on each purchase or sale."
)

if "Open" not in df.columns:
    st.warning("Upload the updated CSV containing opening prices.")

elif len(training_data) < 200:
    st.info("Upload the real historical dataset before backtesting.")

elif st.button("Run backtest"):
    # Use the same chronological split and model settings as before.
    backtest_data = (
        training_data.sort_values("Date").reset_index(drop=True)
    )
    boundary = int(len(backtest_data) * 0.8)

    train_bt = backtest_data.iloc[:boundary - 1].copy()
    test_bt = backtest_data.iloc[boundary:].copy()

    if train_bt["Target"].nunique() < 2:
        st.warning("Training requires both outcome categories.")
        st.stop()

    backtest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )

    backtest_model.fit(
        train_bt[feature_columns],
        train_bt["Target"].astype(int),
    )

    signals = test_bt[["Date"]].copy()
    signals["Position"] = backtest_model.predict(
        test_bt[feature_columns]
    ).astype(int)

    # Build execution prices from the full chronological price history.
    execution = df[["Date"]].copy()
    execution["Entry date"] = df["Date"].shift(-1)
    execution["Entry open"] = df["Open"].shift(-1)
    execution["End date"] = df["Date"].shift(-2)
    execution["End open"] = df["Open"].shift(-2)

    simulation = signals.merge(
        execution,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    # Each measured interval needs two future opens.
    simulation = simulation.dropna(
        subset=["Entry open", "End open"]
    ).reset_index(drop=True)

    if simulation.empty:
        st.warning("Not enough future opening prices to simulate trades.")
        st.stop()

    initial_capital = 10_000.0
    cost_rate = 0.001

    strategy_value = initial_capital
    benchmark_value = initial_capital * (1 - cost_rate)
    previous_position = 0
    order_count = 0
    records = []

    for i, row in simulation.iterrows():
        position = int(row["Position"])
        starting_value = strategy_value
        entry_cost = 0.0
        exit_cost = 0.0

        if position != previous_position:
            action = "BUY" if position == 1 else "SELL"
            entry_cost = strategy_value * cost_rate
            strategy_value -= entry_cost
            order_count += 1
        else:
            action = "HOLD STOCK" if position == 1 else "HOLD CASH"

        market_return = row["End open"] / row["Entry open"] - 1
        market_profit = strategy_value * position * market_return
        strategy_value += market_profit
        benchmark_value *= 1 + market_return

        if i == len(simulation) - 1:
            if position == 1:
                exit_cost = strategy_value * cost_rate
                strategy_value -= exit_cost
                order_count += 1
                action += " + FINAL SELL"
            benchmark_value *= 1 - cost_rate

        total_cost = entry_cost + exit_cost
        net_change = strategy_value - starting_value
        net_return = net_change / starting_value
        expected_value = starting_value + market_profit - total_cost
        if abs(strategy_value - expected_value) > 0.000001:
            raise ValueError("Portfolio accounting does not reconcile.")

        records.append({
            "Prediction date": row["Date"],
            "Entry date": row["Entry date"],
            "End date": row["End date"],
            "Action": action,
            "Position": position,
            "Starting value": starting_value,
            "Market return (%)": market_return * 100,
            "Market profit/loss ($)": market_profit,
            "Trading costs ($)": total_cost,
            "Net change ($)": net_change,
            "Strategy interval return (%)": net_return * 100,
            "Strategy value": strategy_value,
            "Buy-and-hold value": benchmark_value,
        })
        previous_position = position

    backtest_results = pd.DataFrame(records)
    # Include starting capital so early losses are measured correctly.
    portfolio_values = backtest_results.set_index("End date")[
        ["Strategy value", "Buy-and-hold value"]
    ].copy()

    starting_values = pd.DataFrame(
        {
            "Strategy value": [initial_capital],
            "Buy-and-hold value": [initial_capital],
        },
        index=pd.DatetimeIndex(
            [simulation["Entry date"].iloc[0]],
            name="End date",
        ),
    )

    equity_history = pd.concat(
        [starting_values, portfolio_values]
    ).sort_index()

    # Compare each portfolio value with its highest previous value.
    running_peaks = equity_history.cummax()
    drawdowns = (equity_history / running_peaks - 1) * 100

    strategy_worst = drawdowns["Strategy value"].min()
    benchmark_worst = drawdowns["Buy-and-hold value"].min()

    strategy_worst_date = drawdowns["Strategy value"].idxmin()
    benchmark_worst_date = drawdowns["Buy-and-hold value"].idxmin()

    st.subheader("Risk: falls from previous portfolio peaks")

    risk_col1, risk_col2 = st.columns(2)

    risk_col1.metric(
        "Strategy maximum drawdown",
        f"{strategy_worst:.2f}%",
    )
    risk_col2.metric(
        "Buy-and-hold maximum drawdown",
        f"{benchmark_worst:.2f}%",
    )

    drawdown_chart = drawdowns.rename(columns={
        "Strategy value": "Strategy drawdown (%)",
        "Buy-and-hold value": "Buy-and-hold drawdown (%)",
    })

    st.line_chart(drawdown_chart)

    st.write(
        f"Strategy deepest drawdown: "
        f"{strategy_worst_date:%Y-%m-%d}"
    )
    st.write(
        f"Buy-and-hold deepest drawdown: "
        f"{benchmark_worst_date:%Y-%m-%d}"
    )

    stock_exposure = simulation["Position"].mean() * 100

    st.metric(
        "Intervals holding stock",
        f"{stock_exposure:.1f}%",
    )

    st.caption(
        "Drawdowns use the starting balance and recorded interval-end "
        "portfolio values after applicable costs. They do not capture "
        "intraday falls. Exposure is the percentage of simulated "
        "open-to-open intervals spent holding stock."
    )

    # Include drawdowns in the existing simulation table and CSV export.
    backtest_results["Strategy drawdown (%)"] = (
        backtest_results["End date"].map(
            drawdowns["Strategy value"]
        )
    )
    backtest_results["Buy-and-hold drawdown (%)"] = (
        backtest_results["End date"].map(
            drawdowns["Buy-and-hold value"]
        )
    )
    strategy_return = (
        strategy_value / initial_capital - 1
    ) * 100
    benchmark_return = (
        benchmark_value / initial_capital - 1
    ) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Strategy return after costs", f"{strategy_return:.2f}%")
    col2.metric("Buy-and-hold after costs", f"{benchmark_return:.2f}%")
    col3.metric("Strategy buy/sell orders", order_count)

    st.write(
        f"Simulation: {simulation['Entry date'].iloc[0]:%Y-%m-%d}"
        f" to {simulation['End date'].iloc[-1]:%Y-%m-%d}"
    )

    st.line_chart(
        backtest_results.set_index("End date")[
            ["Strategy value", "Buy-and-hold value"]
        ]
    )

    st.subheader("Failure explorer")

    total_market_profit = backtest_results["Market profit/loss ($)"].sum()
    total_costs = backtest_results["Trading costs ($)"].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Market profit/loss earned", f"${total_market_profit:,.2f}")
    col2.metric("Costs deducted", f"${total_costs:,.2f}")
    col3.metric("Net portfolio change", f"${strategy_value - initial_capital:,.2f}")
    st.caption(
        "Market profit/loss earned minus costs deducted equals the net "
        "portfolio change. This is an accounting breakdown of this simulation, "
        "not a separate zero-cost backtest."
    )

    worst_tab, missed_tab, costs_tab = st.tabs([
        "Worst intervals", "Missed gains", "Trading costs",
    ])
    with worst_tab:
        st.write("The 10 intervals with the largest dollar losses, including costs.")
        losing = backtest_results[backtest_results["Net change ($)"] < 0]
        worst = losing.nsmallest(10, "Net change ($)")
        if worst.empty:
            st.info("No losing intervals in this simulation.")
        else:
            st.dataframe(worst[[
                "Prediction date", "Entry date", "End date", "Action",
                "Market return (%)", "Market profit/loss ($)",
                "Trading costs ($)", "Net change ($)",
            ]].round(2), hide_index=True)

    with missed_tab:
        st.write("The 10 largest stock rises during intervals spent in cash.")
        missed = backtest_results[
            (backtest_results["Position"] == 0)
            & (backtest_results["Market return (%)"] > 0)
        ].nlargest(10, "Market return (%)")
        if missed.empty:
            st.info("No rising intervals were spent entirely in cash.")
        else:
            st.dataframe(missed[[
                "Prediction date", "Entry date", "End date", "Action",
                "Market return (%)", "Net change ($)",
            ]].round(2), hide_index=True)
        st.caption(
            "These are missed opportunities, not cash losses. A sale at the "
            "interval start can still incur a cost. Do not add these "
            "percentages together as a total return."
        )

    with costs_tab:
        st.write("Intervals with trading costs, highest first.")
        cost_records = backtest_results[
            backtest_results["Trading costs ($)"] > 0
        ].sort_values("Trading costs ($)", ascending=False)
        st.dataframe(cost_records[[
            "Entry date", "End date", "Action",
            "Trading costs ($)", "Net change ($)",
        ]].round(2), hide_index=True)
        st.caption(
            "Final liquidation is charged at the End date. Other purchases "
            "and sales occur at the Entry date."
        )

    st.subheader("Simulation records")
    st.dataframe(backtest_results.round(2), hide_index=True)

    st.download_button(
        "Download backtest results",
        data=backtest_results.to_csv(index=False).encode("utf-8"),
        file_name="signalguard_backtest.csv",
        mime="text/csv",
    )

    st.caption(
        "Position 1 = holding stock; 0 = holding cash. "
        "Both strategies include entry and final exit costs. "
        "The chart shows values at interval ends."
    )

    st.info(
        "This educational simulation uses adjusted opening prices, "
        "fractional holdings, zero interest on cash, and no taxes. "
        "It measures open-to-open returns, although the model was "
        "trained to predict close-to-close direction. The test period "
        "has already been inspected; this is not new independent evidence."
    )