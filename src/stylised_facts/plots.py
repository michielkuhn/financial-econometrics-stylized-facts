"""Figures for each stylised fact.

Every function takes already-computed data, draws one figure and returns it,
so the plotting layer stays free of any estimation logic.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter, PercentFormatter, ScalarFormatter  # noqa: E402
from scipy import stats  # noqa: E402

from .stats import bartlett_band, sample_acf  # noqa: E402

# Palette: one data hue, one reference hue, neutral ink and grid.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
REFERENCE = "#e34948"
BAND = "#8a8984"
SHADE = "#efeeea"

# NBER US recessions within the sample (peak month, trough month).
US_RECESSIONS = [
    ("1973-11-01", "1975-03-31"),
    ("1980-01-01", "1980-07-31"),
    ("1981-07-01", "1982-11-30"),
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

FREQ_LABEL = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly", "annual": "Annual"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
        }
    )


def _finish(fig, title: str, subtitle: str | None = None) -> None:
    """Lay out the axes, leaving a fixed-height, left-aligned title block on top."""
    pt = 1 / (fig.get_figheight() * 72)  # one typographic point in figure fraction
    fig.tight_layout(rect=(0, 0, 1, 1 - (48 if subtitle else 30) * pt))
    fig.text(0.01, 1 - 8 * pt, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 28 * pt, subtitle, ha="left", va="top", fontsize=10, color=INK_2)


def _pct(ax, axis="y") -> None:
    fmt = PercentFormatter(1.0, decimals=None)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _plain_log(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def _shade_recessions(ax, start, end) -> None:
    for a, b in US_RECESSIONS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b >= start and a <= end:
            ax.axvspan(max(a, start), min(b, end), color=SHADE, lw=0, zorder=0)


def _acf_stems(ax, x, nlags: int, title: str, ylim=None) -> None:
    rho = sample_acf(x, nlags)
    lags = np.arange(1, len(rho) + 1)
    band = bartlett_band(len(x))
    outside = np.abs(rho) > band
    ax.vlines(lags, 0, rho, color=SERIES, lw=1.5)
    ax.scatter(lags[~outside], rho[~outside], s=14, color=SERIES, zorder=3)
    ax.scatter(lags[outside], rho[outside], s=22, color=REFERENCE, zorder=4, label="outside 95% band")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.axhline(band, color=BAND, ls="--", lw=1)
    ax.axhline(-band, color=BAND, ls="--", lw=1)
    ax.set_title(f"{title}\n{outside.sum()} of {len(rho)} lags outside the band", fontsize=10.5)
    ax.set_xlabel("Lag")
    if ylim:
        ax.set_ylim(*ylim)


def _qq(ax, x, dist, title: str) -> None:
    (osm, osr), (slope, intercept, _) = stats.probplot(np.asarray(x), dist=dist)
    ax.scatter(osm, osr, s=8, color=SERIES, alpha=0.7, lw=0)
    ax.plot(osm, slope * osm + intercept, color=REFERENCE, lw=1.5)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")


# --------------------------------------------------------------------------- #
# Overview                                                                     #
# --------------------------------------------------------------------------- #
def price_history(prices: pd.Series, name: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_recessions(ax, prices.index[0], prices.index[-1])
    ax.plot(prices.index, prices, color=SERIES)
    _plain_log(ax)
    ax.set_ylabel("Adjusted close (USD, log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0)
    _finish(
        fig,
        f"{name}: daily adjusted closing price",
        f"{prices.index[0]:%b %Y} – {prices.index[-1]:%b %Y}  ·  shaded bands are NBER US recessions",
    )
    return fig


# --------------------------------------------------------------------------- #
# Fact 1 & 2: stationarity                                                     #
# --------------------------------------------------------------------------- #
def price_nonstationarity(log_prices: pd.Series, name: str, adf: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    lagged = log_prices.shift(1)
    ax1.scatter(lagged, log_prices, s=4, color=SERIES, alpha=0.5, lw=0)
    lo, hi = np.nanmin(log_prices), np.nanmax(log_prices)
    ax1.plot([lo, hi], [lo, hi], color=REFERENCE, ls="--", lw=1.2, label="45° line")
    ax1.set_xlabel(r"$\log P_{t-1}$")
    ax1.set_ylabel(r"$\log P_t$")
    ax1.set_title(r"$\log P_t$ against $\log P_{t-1}$: points sit on the unit-root line", fontsize=10.5)
    ax1.legend(loc="upper left")
    _acf_stems(ax2, log_prices.dropna(), 40, "ACF of daily log price", ylim=(-0.05, 1.05))
    _finish(
        fig,
        "Fact 1 · Prices are non-stationary",
        f"ADF (constant + trend): stat = {adf['ADF stat']:.3f}, p = {adf['p-value']:.3f}  →  unit root not rejected",
    )
    return fig


def return_stationarity(r: pd.Series, name: str, adf: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.plot(r.index, r, color=SERIES, lw=0.5)
    ax1.axhline(0, color=INK_2, lw=0.8)
    _pct(ax1, "y")
    ax1.set_title("Daily log-returns fluctuate around a constant mean", fontsize=10.5)
    ax1.xaxis.set_major_locator(mdates.YearLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.margins(x=0)
    _acf_stems(ax2, r, 40, "ACF of daily log-returns", ylim=(-0.08, 0.08))
    _finish(
        fig,
        "Fact 2 · Returns are stationary",
        f"ADF (no constant, no trend): stat = {adf['ADF stat']:.2f}, "
        f"{'p < 0.0001' if adf['p-value'] < 1e-4 else 'p = %.4f' % adf['p-value']}  →  unit root rejected",
    )
    return fig


# --------------------------------------------------------------------------- #
# Fact 3 & 4: asymmetry and heavy tails                                        #
# --------------------------------------------------------------------------- #
def qq_grid(returns: dict[str, pd.Series], title: str, subtitle: str):
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (freq, r) in zip(axs.flat, returns.items()):
        s = stats.skew(r)
        k = stats.kurtosis(r, fisher=False)
        _qq(ax, r, "norm", f"{FREQ_LABEL[freq]}  ·  skew {s:+.2f}, kurtosis {k:.2f}")
    _finish(fig, title, subtitle)
    return fig


def histograms(returns: dict[str, pd.Series]):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, freq in zip(axs, ["daily", "monthly"]):
        x = np.asarray(returns[freq])
        bins = 120 if freq == "daily" else 40
        ax.hist(x, bins=bins, density=True, color=SERIES, alpha=0.55, edgecolor=SURFACE, lw=0.5, label="Empirical")
        grid = np.linspace(x.min(), x.max(), 400)
        ax.plot(grid, stats.norm.pdf(grid, x.mean(), x.std(ddof=1)), color=REFERENCE, lw=1.8, label="Normal, same mean & sd")
        _pct(ax, "x")
        ax.set_title(f"{FREQ_LABEL[freq]} log-returns", fontsize=10.5)
        ax.set_ylabel("Density")
        if freq == "daily":
            ax.set_xlim(np.quantile(x, 0.0005) * 1.6, np.quantile(x, 0.9995) * 1.6)
        ax.legend(loc="upper left")
    _finish(fig, "Facts 3 & 4 · A taller peak and fatter tails than the normal", "Histograms against a fitted normal density")
    return fig


def qq_student_t(r: pd.Series, dofs=(10, 5, 3)):
    x = np.asarray(r, dtype=float)
    z = (x - x.mean()) / x.std(ddof=1)
    fig, axs = plt.subplots(1, len(dofs) + 1, figsize=(16, 4.4), sharey=True)
    _qq(axs[0], z, "norm", "Normal")
    for ax, nu in zip(axs[1:], dofs):
        _qq(ax, z, stats.t(df=nu), f"Student-t, ν = {nu}")
    for ax in axs[1:]:
        ax.set_ylabel("")
    _finish(
        fig,
        "Fact 4 · Heavy tails: a Student-t with few degrees of freedom fits far better",
        "Standardised daily log-returns against theoretical quantiles",
    )
    return fig


# --------------------------------------------------------------------------- #
# Fact 5: aggregational Gaussianity                                            #
# --------------------------------------------------------------------------- #
def lilliefors_ecdf(full: pd.Series, trimmed: pd.Series, lf_full: tuple, lf_trim: tuple):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, x, (stat, p), label in [
        (axs[0], full, lf_full, "All annual returns"),
        (axs[1], trimmed, lf_trim, "Largest negative year removed"),
    ]:
        xs = np.sort(np.asarray(x))
        n = len(xs)
        ecdf = np.arange(1, n + 1) / n
        grid = np.linspace(xs.min(), xs.max(), 400)
        ax.step(xs, ecdf, where="post", color=SERIES, lw=1.8, label="Empirical CDF")
        ax.plot(grid, stats.norm.cdf(grid, xs.mean(), xs.std(ddof=1)), color=REFERENCE, lw=1.8, label="Fitted normal CDF")
        verdict = "reject normality" if p < 0.05 else "cannot reject normality"
        ax.set_title(f"{label}  ·  D = {stat:.3f}, p = {p:.3f} ({verdict})", fontsize=10.5)
        _pct(ax, "x")
        ax.set_xlabel("Annual log-return")
        ax.legend(loc="upper left")
    axs[0].set_ylabel("Cumulative probability")
    _finish(fig, "Fact 5 · Aggregational Gaussianity", "Lilliefors (Kolmogorov-Smirnov) test on annual log-returns")
    return fig


def kurtosis_by_horizon(norm_full: pd.DataFrame, norm_trim: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    freqs = list(norm_full.columns)
    xpos = np.arange(len(freqs))
    ax.plot(xpos, norm_full.loc["Kurtosis"], marker="o", ms=8, color=SERIES, label="Full sample")
    ax.plot(xpos, norm_trim.loc["Kurtosis"], marker="o", ms=8, color=REFERENCE, label="Most negative return removed")
    ax.axhline(3, color=INK_2, ls="--", lw=1)
    ax.text(-0.1, 3.2, "normal = 3", va="bottom", ha="left", color=INK_2, fontsize=9)
    ax.set_xticks(xpos, [FREQ_LABEL[f] for f in freqs])
    ax.set_ylim(0, None)
    ax.set_ylabel("Kurtosis")
    ax.legend(loc="upper right")
    _finish(fig, "Fact 5 · Kurtosis falls toward 3 as the horizon lengthens")
    return fig


# --------------------------------------------------------------------------- #
# Fact 6 & 7: autocorrelation and volatility clustering                        #
# --------------------------------------------------------------------------- #
def acf_panels(returns: dict[str, pd.Series], squared: bool, title: str, subtitle: str, ylim):
    fig, axs = plt.subplots(1, len(returns), figsize=(16, 4.4), sharey=True)
    for ax, (freq, r) in zip(axs, returns.items()):
        x = np.asarray(r) ** 2 if squared else np.asarray(r)
        label = f"{FREQ_LABEL[freq]}{' squared' if squared else ''}"
        _acf_stems(ax, x, 40, label, ylim=ylim)
    axs[0].set_ylabel("Autocorrelation")
    _finish(fig, title, subtitle)
    return fig


def rolling_volatility(prices: pd.Series, r: pd.Series, name: str, window: int = 252):
    vol = r.rolling(window).std() * np.sqrt(252)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    for ax in (ax1, ax2):
        _shade_recessions(ax, prices.index[0], prices.index[-1])
    ax1.plot(prices.index, prices, color=SERIES)
    _plain_log(ax1)
    ax1.set_ylabel("Adjusted close (log)")
    ax1.set_title("Price", fontsize=10.5)
    ax2.plot(vol.index, vol, color=REFERENCE)
    _pct(ax2, "y")
    ax2.set_ylabel("Annualised volatility")
    ax2.set_title(f"Rolling {window}-day volatility", fontsize=10.5)
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.margins(x=0)
    _finish(
        fig,
        "Facts 7 & 8 · Volatility comes in clusters, and spikes after price falls",
        "Shaded bands are NBER US recessions",
    )
    return fig


# --------------------------------------------------------------------------- #
# Fact 8: leverage effect                                                      #
# --------------------------------------------------------------------------- #
def leverage_scatter(df: pd.DataFrame, corr_neg: float, corr_pos: float):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    neg = df[df["return"] < 0]
    pos = df[df["return"] >= 0]
    ax.scatter(pos["return"], pos["next_month_vol"], s=14, color=BAND, alpha=0.5, lw=0, label=f"Up months (ρ = {corr_pos:+.2f})")
    ax.scatter(neg["return"], neg["next_month_vol"], s=16, color=SERIES, alpha=0.8, lw=0, label=f"Down months (ρ = {corr_neg:+.2f})")
    for sub, color in [(neg, SERIES), (pos, INK_2)]:
        b1, b0 = np.polyfit(sub["return"], sub["next_month_vol"], 1)
        xs = np.linspace(sub["return"].min(), sub["return"].max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=color, lw=2)
    ax.axvline(0, color=INK_2, lw=0.8)
    _pct(ax, "x")
    _pct(ax, "y")
    ax.set_xlabel("Monthly log-return")
    ax.set_ylabel("Realised volatility of the following month")
    ax.legend(loc="upper center")
    _finish(
        fig,
        "Fact 8 · Leverage effect: bigger losses are followed by more volatile months",
        "Next-month volatility = √(Σ squared daily log-returns in month t+1)",
    )
    return fig


def leverage_ccf(ccf: pd.Series, n: int):
    band = bartlett_band(n)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    lags = ccf.index.to_numpy()
    vals = ccf.to_numpy()
    colors = [REFERENCE if (j > 0 and v < -band) else SERIES for j, v in zip(lags, vals)]
    ax.bar(lags, vals, width=0.7, color=colors)
    ax.axhline(band, color=BAND, ls="--", lw=1)
    ax.axhline(-band, color=BAND, ls="--", lw=1)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.axvline(0.5, color=GRID, lw=1)
    ax.set_xlabel("Lag j (days); j > 0: past return vs today's squared return")
    ax.set_ylabel(r"corr$(r_{t-j},\ r_t^2)$")
    n_sig = int(sum((j > 0 and v < -band) for j, v in zip(lags, vals)))
    _finish(
        fig,
        "Fact 8 · Past returns correlate negatively with future squared returns",
        f"Red bars: {n_sig} significantly negative lags among j = 1…{lags.max()} (95% band dashed)",
    )
    return fig
