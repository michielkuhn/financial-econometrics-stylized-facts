"""Empirical tests of the stylised facts of financial returns."""

from .data import load_prices
from .returns import FREQUENCIES, log_returns, returns_by_frequency, drop_min_outlier

__all__ = [
    "load_prices",
    "FREQUENCIES",
    "log_returns",
    "returns_by_frequency",
    "drop_min_outlier",
]
__version__ = "1.0.0"
