"""Quick smoke test for AI Indian Stock Scanner."""

from __future__ import annotations

import traceback


def run_test(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()


def test_imports():
    from data_collection.nse_fetcher import YahooFinanceFetcher
    from analysis.technical_indicators import TechnicalIndicators
    from scanners.momentum_scanner import MomentumScanner

    assert YahooFinanceFetcher
    assert TechnicalIndicators
    assert MomentumScanner


def test_yahoo_fetch():
    from data_collection.nse_fetcher import YahooFinanceFetcher

    fetcher = YahooFinanceFetcher()
    df = fetcher.get_historical_data("RELIANCE", period="1mo", interval="1d")
    assert df is not None


def test_main_app_once_minimal():
    from main import StockScannerApp

    app = StockScannerApp()
    app.symbols = ["RELIANCE", "TCS", "INFY"]
    app.fetch_data()
    results = app.run_scanners()
    assert isinstance(results, list)


if __name__ == "__main__":
    print("Running quick tests...")
    run_test("imports", test_imports)
    run_test("yahoo fetch", test_yahoo_fetch)
    run_test("app minimal once", test_main_app_once_minimal)
    print("Quick tests completed")
