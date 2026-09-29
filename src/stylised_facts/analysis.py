"""End-to-end pipeline: prices in, figures + tables + a Markdown report out."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plots
from .returns import drop_min_outlier, log_returns, returns_by_frequency
from .stats import (
    adf_test,
    anderson_darling,
    cross_correlation,
    leverage_summary,
    monthly_return_vs_next_volatility,
    normality_tests,
    portmanteau_table,
    summary_table,
)

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


@dataclass
class Results:
    name: str
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    verdicts: list[tuple[str, bool, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Formatting helpers                                                           #
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        if float(v).is_integer() and abs(v) >= 1:
            return f"{int(v):,}"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:,.3f}"
    return str(v)


def _p(p: float) -> str:
    # Lilliefors p-values are table-interpolated and floor at 0.001.
    return "< 0.0001" if p < 1e-4 else "≤ 0.001" if p <= 1e-3 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def _peq(p: float) -> str:
    """'p = 0.123' or 'p < 0.0001'."""
    s = _p(p)
    return f"p {s}" if s.startswith("<") else f"p = {s}"


def to_markdown(df: pd.DataFrame, index_label: str = "") -> str:
    cols = [index_label] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + ["--:"] * len(df.columns)) + "|"]
    for idx, row in df.iterrows():
        fmt = _p if "p-value" in str(idx) else _fmt
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def run(prices: pd.Series, name: str, ticker: str, fig_dir: Path, res_dir: Path, lags: int = 40) -> Results:
    fig_dir, res_dir = Path(fig_dir), Path(res_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    plots.apply_style()

    res = Results(name=name, ticker=ticker, start=prices.index[0], end=prices.index[-1])
    log_p = np.log(prices)
    rets = returns_by_frequency(prices)
    rets_trim = {k: drop_min_outlier(v) for k, v in rets.items()}
    daily = rets["daily"]
    crash_day = daily.idxmin()

    # ---- descriptive statistics ------------------------------------------ #
    res.tables["summary_statistics"] = summary_table(rets)
    res.tables["summary_statistics_outlier_removed"] = summary_table(rets_trim)

    # ---- Fact 1 & 2: stationarity ---------------------------------------- #
    adf_price = adf_test(log_p, regression="ct")
    adf_ret = adf_test(daily, regression="n")
    res.tables["adf"] = pd.DataFrame({"log price (const + trend)": adf_price, "daily log-return (none)": adf_ret})
    _save(plots.price_history(prices, name), fig_dir / "01_price_history.png")
    _save(plots.price_nonstationarity(log_p, name, adf_price), fig_dir / "02_fact1_prices_nonstationary.png")
    _save(plots.return_stationarity(daily, name, adf_ret), fig_dir / "03_fact2_returns_stationary.png")
    res.verdicts.append(
        ("1. Prices are non-stationary", adf_price["p-value"] > 0.05,
         f"ADF on log price (constant + trend): stat {adf_price['ADF stat']:.3f}, p = {adf_price['p-value']:.3f}")
    )
    res.verdicts.append(
        ("2. Returns are stationary", adf_ret["p-value"] < 0.05,
         f"ADF on daily log-returns: stat {adf_ret['ADF stat']:.2f}, {_peq(adf_ret['p-value'])}")
    )

    # ---- Fact 3 & 4: asymmetry and heavy tails --------------------------- #
    norm_full = normality_tests(rets)
    norm_trim = normality_tests(rets_trim)
    res.tables["normality"] = norm_full
    res.tables["normality_outlier_removed"] = norm_trim
    _save(
        plots.qq_grid(rets, "Fact 3 · Returns are asymmetric", "Normal QQ plots by sampling frequency, full sample"),
        fig_dir / "04_fact3_qq_by_frequency.png",
    )
    _save(
        plots.qq_grid(
            rets_trim,
            "Fact 3 · Asymmetry without the single worst observation",
            f"Normal QQ plots with the most negative return at each frequency removed (daily: {crash_day:%d %b %Y})",
        ),
        fig_dir / "05_fact3_qq_outlier_removed.png",
    )
    _save(plots.histograms(rets), fig_dir / "06_fact4_histograms.png")
    _save(plots.qq_student_t(daily), fig_dir / "07_fact4_qq_student_t.png")
    skews = norm_full.loc["Skewness"]
    res.verdicts.append(
        ("3. Returns are asymmetric (negatively skewed)", int((skews < 0).sum()) >= 3,
         "Skewness " + ", ".join(f"{k} {v:+.2f}" for k, v in skews.items()))
    )
    kurt = norm_full.loc["Kurtosis"]
    res.verdicts.append(
        ("4. Returns have heavy tails", kurt["daily"] > 3 and norm_full.loc["Jarque-Bera p-value", "daily"] < 0.05,
         f"Daily kurtosis {kurt['daily']:.1f} (normal = 3); Jarque-Bera rejects normality")
    )

    # ---- Fact 5: aggregational Gaussianity ------------------------------- #
    ad_trim = anderson_darling(rets_trim["annual"])
    res.tables["anderson_darling_annual_outlier_removed"] = pd.DataFrame({"annual": ad_trim})
    _save(
        plots.lilliefors_ecdf(
            rets["annual"], rets_trim["annual"],
            (norm_full.loc["Lilliefors stat", "annual"], norm_full.loc["Lilliefors p-value", "annual"]),
            (norm_trim.loc["Lilliefors stat", "annual"], norm_trim.loc["Lilliefors p-value", "annual"]),
        ),
        fig_dir / "08_fact5_lilliefors_annual.png",
    )
    _save(plots.kurtosis_by_horizon(norm_full, norm_trim), fig_dir / "09_fact5_kurtosis_by_horizon.png")
    kurt_trim = norm_trim.loc["Kurtosis"]
    annual_normal = norm_trim.loc["Lilliefors p-value", "annual"] > 0.05 and ad_trim["A2 stat"] < ad_trim["Critical 5%"]
    res.verdicts.append(
        ("5. Aggregational Gaussianity", annual_normal and kurt_trim["annual"] < kurt_trim["daily"],
         f"Kurtosis (worst obs. removed) falls from {kurt_trim['daily']:.1f} daily to {kurt_trim['annual']:.1f} annual; "
         f"annual Lilliefors p = {norm_trim.loc['Lilliefors p-value', 'annual']:.2f}, "
         f"Anderson-Darling A² = {ad_trim['A2 stat']:.3f} < {ad_trim['Critical 5%']:.3f} (5% critical)")
    )

    # ---- Fact 6: no autocorrelation -------------------------------------- #
    port = portmanteau_table(rets, lags=lags)
    res.tables["autocorrelation_returns"] = port
    _save(
        plots.acf_panels(rets, False, "Fact 6 · Returns are (almost) not autocorrelated",
                         f"Sample ACF, lags 1–{lags}, with 95% white-noise bands", ylim=(-0.35, 0.35)),
        fig_dir / "10_fact6_acf_returns.png",
    )
    max_abs = float(np.max(np.abs(port.loc["ACF lag 1"])))
    res.verdicts.append(
        ("6. Returns are not autocorrelated", abs(port.loc["ACF lag 1", "daily"]) < 0.05,
         f"Daily lag-1 ACF {port.loc['ACF lag 1', 'daily']:+.3f}; largest absolute lag-1 ACF at any frequency {max_abs:.3f}. "
         f"Ljung-Box({lags}) p-values: " + ", ".join(f"{k} {_p(v)}" for k, v in port.loc["Ljung-Box p-value"].items()))
    )

    # ---- Fact 7: volatility clustering ----------------------------------- #
    port_sq = portmanteau_table(rets, lags=lags, squared=True)
    res.tables["autocorrelation_squared_returns"] = port_sq
    _save(
        plots.acf_panels(rets, True, "Fact 7 · Squared returns are persistently autocorrelated",
                         f"Sample ACF of squared log-returns, lags 1–{lags}, with 95% white-noise bands", ylim=(-0.35, 0.45)),
        fig_dir / "11_fact7_acf_squared_returns.png",
    )
    _save(plots.rolling_volatility(prices, daily, name), fig_dir / "12_fact7_rolling_volatility.png")
    res.verdicts.append(
        ("7. Volatility clustering", port_sq.loc["Ljung-Box p-value", "daily"] < 0.05,
         f"Daily squared returns: {int(port_sq.loc['Lags outside band', 'daily'])}/{lags} lags outside the band, "
         f"Ljung-Box({lags}) = {port_sq.loc['Ljung-Box stat', 'daily']:,.0f} ({_peq(port_sq.loc['Ljung-Box p-value', 'daily'])})")
    )

    # ---- Fact 8: leverage effect ----------------------------------------- #
    lev = leverage_summary(daily)
    res.tables["leverage_effect"] = pd.DataFrame({"value": lev})
    lev_df = monthly_return_vs_next_volatility(daily)
    ccf = cross_correlation(daily, 20)
    res.tables["cross_correlation"] = ccf.to_frame()
    _save(
        plots.leverage_scatter(lev_df, lev["corr(negative monthly return, next-month vol)"],
                               lev["corr(positive monthly return, next-month vol)"]),
        fig_dir / "13_fact8_leverage_scatter.png",
    )
    _save(plots.leverage_ccf(ccf, len(daily)), fig_dir / "14_fact8_cross_correlation.png")
    n_neg = int(lev["Significant negative lags (1..20)"])
    res.verdicts.append(
        ("8. Leverage effect", lev["corr(negative monthly return, next-month vol)"] < 0 and n_neg > 0,
         f"corr(down-month return, next-month vol) = {lev['corr(negative monthly return, next-month vol)']:+.2f}; "
         f"{n_neg} of 20 lags of corr(r_(t-j), r_t²) significantly negative")
    )

    for key, df in res.tables.items():
        df.to_csv(res_dir / f"{key}.csv")
    return res


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def verdict_table(res: Results) -> str:
    lines = ["| Stylised fact | Supported? | Evidence |", "|:--|:--:|:--|"]
    for fact, ok, evidence in res.verdicts:
        lines.append(f"| {fact} | {'✅' if ok else '❌'} | {evidence} |")
    return "\n".join(lines)


def headline_stats(res: Results) -> str:
    rows = ["Mean (%)", "Std. deviation (%)", "Skewness", "Kurtosis", "Min (%)", "Max (%)",
            "Jarque-Bera p-value", "Lilliefors p-value", "Observations"]
    return to_markdown(res.tables["summary_statistics"].loc[rows], "Log-returns")


def write_report(res: Results, path: Path, fig_rel: str = "../figures") -> None:
    t = res.tables
    period = f"{res.start:%d %b %Y} – {res.end:%d %b %Y}"
    parts = [
        f"# Stylised facts of {res.name} ({res.ticker}) returns: full results",
        f"_Auto-generated by `run_analysis.py` · sample {period}_",
        "## Verdicts", verdict_table(res),
        "## Summary statistics (full sample)", to_markdown(t["summary_statistics"], "Statistic"),
        "## Summary statistics (most negative observation removed at each frequency)",
        to_markdown(t["summary_statistics_outlier_removed"], "Statistic"),
        "## Augmented Dickey-Fuller tests", to_markdown(t["adf"], ""),
        "## Normality tests", to_markdown(t["normality"], ""),
        "### Most negative observation removed", to_markdown(t["normality_outlier_removed"], ""),
        "### Anderson-Darling, annual returns without the worst year",
        to_markdown(t["anderson_darling_annual_outlier_removed"], ""),
        "## Autocorrelation of returns", to_markdown(t["autocorrelation_returns"], ""),
        "## Autocorrelation of squared returns", to_markdown(t["autocorrelation_squared_returns"], ""),
        "## Leverage effect", to_markdown(t["leverage_effect"], ""),
        "## Figures",
    ]
    figs = sorted(Path(path).parent.parent.joinpath("figures").glob("*.png"))
    parts += [f"![{f.stem}]({fig_rel}/{f.name})" for f in figs]
    Path(path).write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def update_readme(res: Results, readme: Path) -> bool:
    """Replace the block between the RESULTS markers with fresh numbers."""
    readme = Path(readme)
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    period = f"{res.start:%d %b %Y} – {res.end:%d %b %Y}"
    block = "\n\n".join([
        README_START,
        f"_Sample: {period}. Numbers below are regenerated by `python run_analysis.py`._",
        verdict_table(res),
        headline_stats(res),
        README_END,
    ])
    new = re.sub(re.escape(README_START) + ".*?" + re.escape(README_END), lambda _: block, text, flags=re.S)
    readme.write_text(new, encoding="utf-8")
    return True
