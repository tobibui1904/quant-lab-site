import os
import sys
import numpy as np
import pandas as pd
import duckdb
from dotenv import load_dotenv

from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.model_selection import GridSearchCV, train_test_split
from skfolio import Population, RatioMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.distance import KendallDistance, PearsonDistance
from skfolio.metrics import make_scorer
from skfolio.model_selection import WalkForward, cross_val_predict, optimal_folds_number, CombinatorialPurgedCV
from skfolio.optimization import (
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
)
from skfolio.preprocessing import prices_to_returns

np.random.seed(123)

# When stdout is a pipe rather than a console -- as under the hub, whose output
# is redirected to hub.log -- Python encodes it with the locale codec, cp1252 on
# this machine, and the check marks and arrows data() prints raise
# UnicodeEncodeError. Force UTF-8 on the real streams; marimo's in-notebook
# capture stream has no reconfigure().
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# --- Alpaca client ---

def _get_api():
    # Credentials come from .env only (see .env.example). There is deliberately
    # no hardcoded fallback: the previous defaults were real keys sitting in a
    # tracked file, and — because the secret was read from ALPACA_SECRET while
    # every other module writes ALPACA_API_SECRET — the env override never
    # matched and this always authenticated with the baked-in value.
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Copy .env.example to .env and set "
            "ALPACA_API_KEY and ALPACA_API_SECRET."
        )
    return REST(api_key, api_secret, base_url)


# --- Pipeline functions ---

def data(pair_file="pair.txt", start_date="2019-01-01"):
    api = _get_api()
    tickers = ["AMZN", "AAPL", "MSFT", "GOOG", "NVDA", "TSM", "IBM"]

    if os.path.exists(pair_file):
        with open(pair_file, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
            for ticker in last_line.strip().split(" "):
                if ticker:
                    tickers.append(ticker)

    # The selected pair can overlap the fixed universe above. dict.fromkeys
    # dedupes while preserving order.
    tickers = list(dict.fromkeys(tickers))

    print(f"Attempting to load {len(tickers)} tickers: {tickers}")

    # Symbols Alpaca cannot trade (preferred lines like "BEP/PA") would fail
    # the request for everything else alongside them.
    tickers = [t for t in tickers if "/" not in t]

    # One multi-symbol request instead of one request per ticker. Alpaca fails
    # the WHOLE request if any symbol in it is unknown, so bisect on failure to
    # isolate the bad names rather than losing every good ticker with them.
    frames, failed = [], []

    def _fetch(symbols):
        if not symbols:
            return
        try:
            df = api.get_bars(symbols, TimeFrame.Day, start=start_date).df
            if not df.empty:
                frames.append(df[["symbol", "close"]])
            return
        except Exception as e:
            if len(symbols) == 1:
                failed.append(symbols[0])
                print(f"  ✗ {symbols[0]}: {e}")
                return
            mid = len(symbols) // 2
            _fetch(symbols[:mid])
            _fetch(symbols[mid:])

    _fetch(tickers)

    if not frames:
        raise RuntimeError(
            f"Alpaca returned no bars for any of {tickers}. Rejected: {failed}"
        )

    bars = pd.concat(frames)
    bars.index = bars.index.tz_localize(None)
    # Long -> wide. pivot_table with "last" tolerates the duplicate
    # (timestamp, symbol) rows that pagination can hand back; plain pivot raises.
    close_prices = bars.pivot_table(
        index=bars.index, columns="symbol", values="close", aggfunc="last"
    )
    close_prices.index.name = "timestamp"

    for ticker in tickers:
        if ticker in close_prices.columns:
            col = close_prices[ticker].dropna()
            print(f"  ✓ {ticker}: {len(col)} bars "
                  f"({col.index[0].date()} → {col.index[-1].date()})")
        else:
            print(f"  ✗ {ticker}: no data returned")

    print(f"\nBefore dropna: {close_prices.shape}")
    print(f"NaN counts per ticker:\n{close_prices.isna().sum()}")

    # Drop rows (dates) with any missing data — keeps all assets
    close_prices = close_prices.dropna(axis=0)

    print(f"After dropna:  {close_prices.shape} "
          f"({close_prices.index[0].date()} → {close_prices.index[-1].date()})")

    X = prices_to_returns(close_prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)
    return X, X_train, X_test


def portfolio_model(X_train):
    model_hrp = HierarchicalRiskParity(
        risk_measure=RiskMeasure.CVAR,
        hierarchical_clustering_estimator=HierarchicalClustering(),
    )
    model_herc = HierarchicalEqualRiskContribution(
        risk_measure=RiskMeasure.CVAR,
        hierarchical_clustering_estimator=HierarchicalClustering(),
    )

    # WalkForward is sklearn-compatible → use for GridSearchCV
    cv_wf = WalkForward(train_size=252, test_size=9)

    grid_search = GridSearchCV(
        estimator=model_hrp,
        cv=cv_wf,
        n_jobs=-1,
        param_grid={
            "distance_estimator": [PearsonDistance(), KendallDistance()],
            "hierarchical_clustering_estimator__linkage_method": [
                LinkageMethod.SINGLE,
                LinkageMethod.WARD,
                LinkageMethod.COMPLETE,
            ],
        },
        scoring=make_scorer(RatioMeasure.CVAR_RATIO),
    )
    grid_search.fit(X_train)
    model_hrp = grid_search.best_estimator_

    grid_search.set_params(estimator=model_herc).fit(X_train)
    model_herc = grid_search.best_estimator_

    # CPCV for final cross_val_predict evaluation
    n_folds, n_test_folds = optimal_folds_number(
        n_observations=X_train.shape[0],
        target_n_test_paths=100,
        target_train_size=252,
    )
    cv_cpcv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds)

    return model_hrp, model_herc, cv_cpcv


def cross_predict(model_hrp, model_herc, X_test, cv):
    pred_hrp = cross_val_predict(
        model_hrp, X_test, cv=cv, n_jobs=-1, portfolio_params=dict(name="HRP", tag = "HRP")
    )
    pred_herc = cross_val_predict(
        model_herc, X_test, cv=cv, n_jobs=-1, portfolio_params=dict(name="HERC", tag = "HERC")
    )
    # population = Population([pred_hrp, pred_herc])
    population = pred_hrp + pred_herc
    return population, pred_hrp, pred_herc


def statistics_report(population):
    return population.summary()

def portfolio_ratio_duckdb(model, db_path="quant_trading.db"):
    # If model is a Population (CPCV result), pick the best portfolio by CVaR ratio
    if isinstance(model, Population):
        model = model.max_measure(RatioMeasure.CVAR_RATIO)

    weights = model.weights_per_observation.iloc[-1].to_dict()
    date    = pd.to_datetime(model.weights_per_observation.iloc[-1].name)

    with duckdb.connect(db_path) as con:
        result = con.execute("SELECT MAX(id) FROM Portfolio").fetchone()
        last_id = result[0] if result[0] is not None else 0
        new_id  = last_id + 1

        existing_cols_info = con.execute("PRAGMA table_info('Portfolio')").fetchall()
        existing_cols      = [row[1] for row in existing_cols_info]
        existing_cols_lower = set(c.lower() for c in existing_cols)

        for col in weights:
            if col.lower() not in existing_cols_lower:
                con.execute(f'ALTER TABLE Portfolio ADD COLUMN "{col}" DOUBLE')
                existing_cols.append(col)
                existing_cols_lower.add(col.lower())

        if "Date" not in existing_cols:
            con.execute("ALTER TABLE Portfolio ADD COLUMN Date TIMESTAMP")
            existing_cols.append("Date")

        row = {col: None for col in existing_cols}
        row["id"]   = new_id
        row["Date"] = date
        row.update(weights)

        df = pd.DataFrame([row])[existing_cols].fillna(0)
        con.register("temp_portfolio", df)
        con.execute("INSERT INTO Portfolio SELECT * FROM temp_portfolio")
        con.unregister("temp_portfolio")