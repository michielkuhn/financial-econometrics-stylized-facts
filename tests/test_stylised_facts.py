import numpy as np
import pandas as pd
import pytest

from stylised_facts.analysis import README_END, README_START, run, update_readme, write_report
from stylised_facts.data import read_price_csv
from stylised_facts.returns import drop_min_outlier, log_returns, returns_by_frequency
from stylised_facts.stats import (
    adf_test,
    bartlett_band,
    cross_correlation,
    leverage_summary,
    monthly_return_vs_next_volatility,
    portmanteau_table,
    summary_statistics,
)


def test_lower_frequency_returns_aggregate_daily_returns(prices):
    daily = log_returns(prices, "daily")
    monthly = log_returns(prices, "monthly")
    # Each full month's log-return equals the sum of its daily log-returns.
    summed = daily.resample("ME").sum().iloc[1:]
    pd.testing.assert_series_equal(monthly, summed.loc[monthly.index], check_names=False)


def test_unknown_frequency_raises(prices):
    with pytest.raises(ValueError):
        log_returns(prices, "hourly")


def test_drop_min_outlier_removes_exactly_one(prices):
    r = log_returns(prices)
    trimmed = drop_min_outlier(r)
    assert len(trimmed) == len(r) - 1
    assert trimmed.min() > r.min()


def test_summary_statistics_on_normal_data():
    x = pd.Series(np.random.default_rng(1).normal(0, 0.01, 20_000))
    s = summary_statistics(x)
    assert s["Kurtosis"] == pytest.approx(3, abs=0.15)
    assert s["Excess kurtosis"] == pytest.approx(0, abs=0.15)
    assert s["Std. deviation (%)"] == pytest.approx(1.0, rel=0.03)
    assert s["Jarque-Bera p-value"] > 0.01
    assert s["Observations"] == 20_000


def test_adf_distinguishes_random_walk_from_noise():
    rng = np.random.default_rng(2)
    eps = rng.normal(size=3000)
    assert adf_test(np.cumsum(eps), "ct")["p-value"] > 0.05
    assert adf_test(eps, "n")["p-value"] < 0.01


def test_simulated_garch_shows_stylised_facts(prices):
    daily = log_returns(prices)
    assert summary_statistics(daily)["Kurtosis"] > 4
    sq = portmanteau_table({"daily": daily}, squared=True)
    assert sq.loc["Ljung-Box p-value", "daily"] < 0.01
    lev = leverage_summary(daily)
    assert lev["corr(r_{t-1}, r_t^2)"] < 0


def test_cross_correlation_lag_convention():
    # Volatility today rises after a negative return yesterday: only lag +1 should react.
    rng = np.random.default_rng(3)
    n = 20_000
    r = np.empty(n)
    r[0] = rng.normal()
    for t in range(1, n):
        r[t] = rng.normal() * (1.5 if r[t - 1] < 0 else 0.5)
    ccf = cross_correlation(pd.Series(r), 3)
    assert list(ccf.index) == [-3, -2, -1, 0, 1, 2, 3]
    assert ccf.loc[1] < -0.2
    assert abs(ccf.loc[-1]) < 0.05


def test_next_month_volatility_is_not_contemporaneous(prices):
    daily = log_returns(prices)
    df = monthly_return_vs_next_volatility(daily)
    month = df.index[5]
    nxt = daily[(daily.index > month) & (daily.index <= month + pd.offsets.MonthEnd(1))]
    assert df.loc[month, "next_month_vol"] == pytest.approx(np.sqrt((nxt**2).sum()))


def test_bartlett_band():
    assert bartlett_band(400) == pytest.approx(1.96 / 20, rel=1e-3)


def test_csv_roundtrip(tmp_path, prices):
    path = tmp_path / "p.csv"
    prices.rename_axis("Date").to_csv(path)
    back = read_price_csv(path)
    np.testing.assert_allclose(back.values, prices.values)


def test_pipeline_end_to_end(tmp_path, prices):
    figs, res_dir = tmp_path / "figures", tmp_path / "results"
    res = run(prices, "Simulated", "SIM", figs, res_dir)
    assert len(res.verdicts) == 8
    assert len(list(figs.glob("*.png"))) == 14
    assert (res_dir / "summary_statistics.csv").exists()
    write_report(res, res_dir / "REPORT.md")
    assert "Verdicts" in (res_dir / "REPORT.md").read_text()

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n")
    assert update_readme(res, readme)
    text = readme.read_text()
    assert "old" not in text and "intro" in text and "outro" in text and "Leverage effect" in text


def test_returns_by_frequency_keys(prices):
    assert list(returns_by_frequency(prices)) == ["daily", "weekly", "monthly", "annual"]
