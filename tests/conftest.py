import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def simulate_gjr_garch_prices(n: int = 6000, seed: int = 0) -> pd.Series:
    """Prices from a GJR-GARCH(1,1) with Student-t shocks.

    The process is built to exhibit the stylised facts (heavy tails, volatility
    clustering, leverage), so the tests can check the detectors find them.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, gamma, beta, nu = 2e-6, 0.03, 0.08, 0.90, 5
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    r = np.empty(n)
    h = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        r[t] = 3e-4 + np.sqrt(h) * z[t]
        h = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * h
    dates = pd.bdate_range("1990-01-01", periods=n)
    return pd.Series(20 * np.exp(np.cumsum(r)), index=dates, name="Adj Close")


@pytest.fixture(scope="session")
def prices() -> pd.Series:
    return simulate_gjr_garch_prices()
