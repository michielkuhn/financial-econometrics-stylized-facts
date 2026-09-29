#!/usr/bin/env python
"""Run the full stylised-facts analysis.

Examples
--------
    python run_analysis.py                              # Coca-Cola, 1973-2023
    python run_analysis.py --ticker SPY --name "S&P 500 ETF" --start 1994-01-01
    python run_analysis.py --csv my_prices.csv          # offline, your own data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stylised_facts.analysis import run, update_readme, verdict_table, write_report  # noqa: E402
from stylised_facts.data import load_prices  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="KO")
    p.add_argument("--name", default="Coca-Cola")
    p.add_argument("--start", default="1973-12-31")
    p.add_argument("--end", default="2023-12-31")
    p.add_argument("--csv", help="CSV with Date and Adj Close columns (skips the download)")
    p.add_argument("--refresh", action="store_true", help="ignore the local cache and re-download")
    p.add_argument("--figures", default=str(ROOT / "figures"))
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--no-readme", action="store_true", help="do not update the results block in README.md")
    args = p.parse_args()

    prices = load_prices(args.ticker, args.start, args.end, cache_dir=ROOT / "data", csv=args.csv, refresh=args.refresh)
    print(f"Loaded {len(prices):,} daily prices for {args.ticker}: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")

    res = run(prices, args.name, args.ticker, Path(args.figures), Path(args.results))
    write_report(res, Path(args.results) / "REPORT.md")
    if not args.no_readme and update_readme(res, ROOT / "README.md"):
        print("Updated results block in README.md")

    print()
    print(verdict_table(res))
    print(f"\nFigures -> {args.figures}\nTables  -> {args.results}")


if __name__ == "__main__":
    main()
