"""Statistical tests used to assess each stylised fact."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, lilliefors
from statsmodels.tsa.stattools import acf, adfuller


# --------------------------------------------------------------------------- #
# Descriptive statistics                                                       #
# --------------------------------------------------------------------------- #
def summary_statistics(r: pd.Series) -> dict[str, float]:
    """Moments, quantiles and normality tests for one return series.

    Location and scale measures are reported in percent. ``Kurtosis`` is the
    raw fourth standardised moment (3 for a normal distribution).
    """
    x = np.asarray(r, dtype=float)
    n = len(x)
    kurt = stats.kurtosis(x, fisher=False)
    jb = stats.jarque_bera(x)
    lf_stat, lf_p = lilliefors(x, dist="norm", pvalmethod="table")
    q = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "Mean (%)": 100 * x.mean(),
        "Std. deviation (%)": 100 * x.std(ddof=1),
        "95% CI half-width of mean (%)": 100 * 1.96 * x.std(ddof=1) / np.sqrt(n),
        "Skewness": stats.skew(x),
        "Kurtosis": kurt,
        "Excess kurtosis": kurt - 3,
        "Min (%)": 100 * x.min(),
        "5% quantile (%)": 100 * q[0],
        "25% quantile (%)": 100 * q[1],
        "Median (%)": 100 * q[2],
        "75% quantile (%)": 100 * q[3],
        "95% quantile (%)": 100 * q[4],
        "Max (%)": 100 * x.max(),
        "Jarque-Bera stat": jb.statistic,
        "Jarque-Bera p-value": jb.pvalue,
        "Lilliefors stat": lf_stat,
        "Lilliefors p-value": lf_p,
        "Observations": n,
    }


def summary_table(returns: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({name: summary_statistics(r) for name, r in returns.items()})


# --------------------------------------------------------------------------- #
# Stationarity                                                                 #
# --------------------------------------------------------------------------- #
def adf_test(x: pd.Series, regression: str = "c") -> dict[str, float]:
    """Augmented Dickey-Fuller test; H0: the series has a unit root.

    ``regression`` is ``"ct"`` (constant + trend) for log prices and ``"n"``
    (no deterministic terms) for returns, as motivated in the README.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # tuple-return deprecation in statsmodels >= 0.15
        stat, pvalue, usedlag, nobs, crit, _ = adfuller(np.asarray(x, dtype=float), regression=regression, autolag="AIC")
    return {
        "ADF stat": stat,
        "p-value": pvalue,
        "Lags used": usedlag,
        "Observations": nobs,
        "Critical 1%": crit["1%"],
        "Critical 5%": crit["5%"],
        "Critical 10%": crit["10%"],
    }


# --------------------------------------------------------------------------- #
# Normality                                                                    #
# --------------------------------------------------------------------------- #
def _anderson(x):
    # SciPy >= 1.17 warns that critical-value output will change; we rely on it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return stats.anderson(np.asarray(x, dtype=float), dist="norm")


def anderson_darling(x: pd.Series) -> dict[str, float]:
    """Anderson-Darling normality test with critical values by significance level."""
    res = _anderson(x)
    out = {"A2 stat": res.statistic}
    for level, cv in zip(res.significance_level, res.critical_values):
        out[f"Critical {level:g}%"] = cv
    return out


def normality_tests(returns: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, r in returns.items():
        x = np.asarray(r, dtype=float)
        jb = stats.jarque_bera(x)
        lf_stat, lf_p = lilliefors(x, dist="norm", pvalmethod="table")
        ad = _anderson(x)
        ad_5 = ad.critical_values[list(ad.significance_level).index(5.0)]
        rows[name] = {
            "Skewness": stats.skew(x),
            "Kurtosis": stats.kurtosis(x, fisher=False),
            "Jarque-Bera stat": jb.statistic,
            "Jarque-Bera p-value": jb.pvalue,
            "Lilliefors stat": lf_stat,
            "Lilliefors p-value": lf_p,
            "Anderson-Darling A2": ad.statistic,
            "A-D critical 5%": ad_5,
            "Observations": len(x),
        }
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Serial dependence                                                            #
# --------------------------------------------------------------------------- #
def bartlett_band(n: int, level: float = 0.95) -> float:
    """Half-width of the white-noise confidence band for sample autocorrelations."""
    return stats.norm.ppf(0.5 + level / 2) / np.sqrt(n)


def sample_acf(x: pd.Series, nlags: int = 40) -> np.ndarray:
    """Autocorrelations at lags 1..nlags (lag 0 dropped)."""
    x = np.asarray(x, dtype=float)
    return acf(x, nlags=min(nlags, len(x) - 1), fft=True)[1:]


def portmanteau_table(returns: dict[str, pd.Series], lags: int = 40, squared: bool = False) -> pd.DataFrame:
    """Lag-1 ACF test plus Box-Pierce and Ljung-Box tests up to ``lags``."""
    rows = {}
    for name, r in returns.items():
        x = np.asarray(r, dtype=float)
        if squared:
            x = x**2
        n = len(x)
        h = min(lags, n - 1)
        rho = sample_acf(x, h)
        band = bartlett_band(n)
        lb = acorr_ljungbox(x, lags=[h], boxpierce=True)
        rows[name] = {
            "ACF lag 1": rho[0],
            "95% band": band,
            "z (lag 1)": rho[0] * np.sqrt(n),
            "Lags outside band": int(np.sum(np.abs(rho) > band)),
            "Lags tested": h,
            "Box-Pierce stat": lb["bp_stat"].iloc[0],
            "Box-Pierce p-value": lb["bp_pvalue"].iloc[0],
            "Ljung-Box stat": lb["lb_stat"].iloc[0],
            "Ljung-Box p-value": lb["lb_pvalue"].iloc[0],
            "Chi2 critical 5%": stats.chi2.ppf(0.95, h),
        }
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Leverage effect                                                              #
# --------------------------------------------------------------------------- #
def cross_correlation(r: pd.Series, max_lag: int = 20) -> pd.Series:
    """corr(r_{t-j}, r_t^2) for j = -max_lag..max_lag.

    Positive ``j`` means past returns versus current squared returns; a
    negative value there is the leverage effect.
    """
    x = pd.Series(np.asarray(r, dtype=float))
    x2 = x**2
    out = {j: x.shift(j).corr(x2) for j in range(-max_lag, max_lag + 1)}
    return pd.Series(out, name="ccf")


def monthly_return_vs_next_volatility(daily_returns: pd.Series) -> pd.DataFrame:
    """Monthly log-return against realised volatility of the *following* month.

    Realised volatility is the square root of the sum of squared daily
    log-returns within the month, so it does not share any observation with the
    return it is compared to.
    """
    monthly_ret = daily_returns.resample("ME").sum()
    realised_vol = np.sqrt((daily_returns**2).resample("ME").sum())
    df = pd.DataFrame({"return": monthly_ret, "next_month_vol": realised_vol.shift(-1)})
    return df.dropna()


def leverage_summary(daily_returns: pd.Series, max_lag: int = 20) -> dict[str, float]:
    df = monthly_return_vs_next_volatility(daily_returns)
    neg = df[df["return"] < 0]
    pos = df[df["return"] >= 0]
    ccf = cross_correlation(daily_returns, max_lag)
    band = bartlett_band(len(daily_returns))
    past = ccf.loc[1:]
    slope_neg = np.polyfit(neg["return"], neg["next_month_vol"], 1)[0]
    return {
        "corr(negative monthly return, next-month vol)": neg["return"].corr(neg["next_month_vol"]),
        "corr(positive monthly return, next-month vol)": pos["return"].corr(pos["next_month_vol"]),
        "OLS slope, negative months": slope_neg,
        "corr(r_{t-1}, r_t^2)": ccf.loc[1],
        "Significant negative lags (1..%d)" % max_lag: int((past < -band).sum()),
        "Significant positive lags (1..%d)" % max_lag: int((past > band).sum()),
        "95% band": band,
    }
