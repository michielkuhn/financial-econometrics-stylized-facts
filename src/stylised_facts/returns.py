"""Log-returns at several sampling frequencies."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Pandas resampling rules; weeks end on Friday so each bar is a trading week.
FREQUENCIES: dict[str, str | None] = {
    "daily": None,
    "weekly": "W-FRI",
    "monthly": "ME",
    "annual": "YE",
}


def log_returns(prices: pd.Series, freq: str = "daily") -> pd.Series:
    """Continuously compounded returns ``r_t = log(P_t) - log(P_{t-1})``.

    Lower frequencies use the last available price in each period, so they are
    exact sums of the daily log-returns within that period.
    """
    if freq not in FREQUENCIES:
        raise ValueError(f"freq must be one of {list(FREQUENCIES)}, got {freq!r}")
    log_p = np.log(prices)
    rule = FREQUENCIES[freq]
    if rule is not None:
        log_p = log_p.resample(rule).last().dropna()
    r = log_p.diff().dropna()
    r.name = f"{freq} log-return"
    return r


def returns_by_frequency(prices: pd.Series) -> dict[str, pd.Series]:
    return {freq: log_returns(prices, freq) for freq in FREQUENCIES}


def drop_min_outlier(r: pd.Series) -> pd.Series:
    """Remove the single most negative observation (e.g. the 1987 crash)."""
    return r.drop(r.idxmin())
