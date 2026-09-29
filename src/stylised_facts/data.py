"""Price data loading with a local CSV cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    return cache_dir / f"{ticker}_{start}_{end}.csv"


def read_price_csv(path: str | Path) -> pd.Series:
    """Read a CSV with a date column and an adjusted-close column.

    Accepts the layout written by this package (``Date,Adj Close``) as well as
    common exports that name the columns ``date``/``adj_close``/``close``.
    """
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in {"date", "datetime", "timestamp"})
    candidates = ["Adj Close", "adj_close", "adjclose", "Adj_Close", "Close", "close"]
    price_col = next(c for c in candidates if c in df.columns)
    s = pd.Series(df[price_col].to_numpy(dtype=float), index=pd.to_datetime(df[date_col]))
    s.index.name = "Date"
    s.name = "Adj Close"
    return s.sort_index().dropna()


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download dividend- and split-adjusted daily closing prices from Yahoo Finance."""
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker!r} between {start} and {end}.")
    adj = raw["Adj Close"]
    # Recent yfinance versions return a (field, ticker) MultiIndex even for one ticker.
    if isinstance(adj, pd.DataFrame):
        adj = adj.iloc[:, 0]
    adj = adj.astype(float).dropna()
    adj.index = pd.to_datetime(adj.index).tz_localize(None)
    adj.index.name = "Date"
    adj.name = "Adj Close"
    return adj


def load_prices(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path = "data",
    csv: str | Path | None = None,
    refresh: bool = False,
) -> pd.Series:
    """Return daily adjusted closing prices.

    Order of precedence: an explicit ``csv`` file, then the local cache, then a
    fresh download (which is written to the cache).
    """
    if csv is not None:
        return read_price_csv(csv).loc[start:end]

    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, ticker, start, end)
    if path.exists() and not refresh:
        return read_price_csv(path)

    prices = download_prices(ticker, start, end)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    return prices
