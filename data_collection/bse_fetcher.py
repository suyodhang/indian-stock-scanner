"""BSE data fetcher and scanner."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)


BSE_STOCK_CODES: Dict[str, str] = {
    "RELIANCE": "500325",
    "TCS": "532540",
    "HDFCBANK": "500180",
    "INFY": "500209",
    "ICICIBANK": "532174",
    "SBIN": "500112",
    "ITC": "500875",
    "LT": "500510",
    "HCLTECH": "532281",
    "AXISBANK": "532215",
    "ASIANPAINT": "500820",
    "TATAMOTORS": "500570",
    "WIPRO": "507685",
    "TATASTEEL": "500470",
    "ONGC": "500312",
    "YESBANK": "532648",
    "ZOMATO": "543320",
    "IRCTC": "542830",
    "IRFC": "543257",
    "PAYTM": "543396",
}

BSE_INDEX_CODES: Dict[str, str] = {
    "SENSEX": "16",
    "BSE100": "22",
    "BSE200": "23",
    "BSE500": "17",
    "MIDCAP": "24",
    "SMALLCAP": "25",
    "BANKEX": "53",
    "IT": "50",
    "HEALTHCARE": "54",
    "AUTO": "42",
    "FMCG": "45",
    "METAL": "47",
    "OIL&GAS": "51",
    "POWER": "69",
    "REALTY": "59",
    "TELECOM": "52",
    "ENERGY": "57",
    "PSU": "72",
}


@dataclass
class BSEStockInfo:
    scrip_code: str
    symbol: str
    company_name: str
    group: str = ""
    face_value: float = 0.0
    isin: str = ""
    industry: str = ""


@dataclass
class BSEQuote:
    scrip_code: str
    symbol: str
    company_name: str
    current_price: float
    change: float
    change_pct: float
    open_price: float
    high: float
    low: float
    close: float
    prev_close: float
    volume: int
    value: float
    market_cap: float
    total_buy_qty: int
    total_sell_qty: int
    upper_circuit: float
    lower_circuit: float
    timestamp: str


class BSESession:
    BASE_URL = "https://api.bseindia.com"
    WEBSITE_URL = "https://www.bseindia.com"

    def __init__(self, max_retries: int = 3, delay: float = 0.4):
        self.session = requests.Session()
        self.max_retries = max_retries
        self.delay = delay
        self.last_request = 0.0
        self._setup_session()

    def _setup_session(self) -> None:
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Referer": self.WEBSITE_URL,
                "Origin": self.WEBSITE_URL,
            }
        )
        try:
            self.session.get(self.WEBSITE_URL, timeout=10)
        except Exception:
            pass

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Optional[requests.Response]:
        self._rate_limit()
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (403, 429):
                    time.sleep(1 + attempt)
                    self._setup_session()
            except requests.RequestException:
                time.sleep(1 + attempt)
        return None

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        resp = self.get(url, params=params)
        if not resp:
            return None
        try:
            return resp.json()
        except json.JSONDecodeError:
            return None


class BSEFetcher:
    def __init__(self):
        self.session = BSESession()
        self.base_url = self.session.BASE_URL

    # STOCK CODE UTILITIES
    def get_scrip_code(self, symbol: str) -> Optional[str]:
        sym = symbol.upper()
        if sym in BSE_STOCK_CODES:
            return BSE_STOCK_CODES[sym]
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/Suggest/BSESuggest", {"query": sym})
        if isinstance(data, list) and data:
            parts = str(data[0]).split("/")
            return parts[0].strip() if parts else None
        return None

    def get_symbol_from_code(self, scrip_code: str) -> Optional[str]:
        for symbol, code in BSE_STOCK_CODES.items():
            if code == str(scrip_code):
                return symbol
        return None

    def search_stocks(self, query: str) -> List[Dict[str, str]]:
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/Suggest/BSESuggest", {"query": query})
        out: List[Dict[str, str]] = []
        if isinstance(data, list):
            for item in data:
                parts = str(item).split("/")
                if len(parts) >= 3:
                    out.append({"scrip_code": parts[0], "symbol": parts[1], "company_name": parts[2]})
        return out

    # REAL-TIME DATA
    def get_stock_quote(self, scrip_code_or_symbol: str) -> Optional[BSEQuote]:
        scrip = str(scrip_code_or_symbol)
        if not scrip.isdigit():
            scrip = self.get_scrip_code(scrip) or scrip
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/getScripHeaderData/Equity/{scrip}")
        header = data.get("Header", {}) if isinstance(data, dict) else {}
        if not header:
            return None
        return BSEQuote(
            scrip_code=str(scrip),
            symbol=header.get("ScripID", ""),
            company_name=header.get("CompanyName", ""),
            current_price=self._safe_float(header.get("CurrRate", 0)),
            change=self._safe_float(header.get("Change", 0)),
            change_pct=self._safe_float(header.get("PcChange", 0)),
            open_price=self._safe_float(header.get("Open", 0)),
            high=self._safe_float(header.get("High", 0)),
            low=self._safe_float(header.get("Low", 0)),
            close=self._safe_float(header.get("CurrRate", 0)),
            prev_close=self._safe_float(header.get("PrevClose", 0)),
            volume=self._safe_int(header.get("TotalVol", 0)),
            value=self._safe_float(header.get("TotalVal", 0)),
            market_cap=self._safe_float(header.get("MktCap", 0)),
            total_buy_qty=self._safe_int(header.get("TotalBuyQty", 0)),
            total_sell_qty=self._safe_int(header.get("TotalSellQty", 0)),
            upper_circuit=self._safe_float(header.get("UprCktLm", 0)),
            lower_circuit=self._safe_float(header.get("LwrCktLm", 0)),
            timestamp=header.get("DispTime", datetime.now().strftime("%H:%M:%S")),
        )

    def get_stock_quote_by_symbol(self, symbol: str) -> Optional[BSEQuote]:
        return self.get_stock_quote(symbol)

    def get_multiple_quotes(self, scrip_codes: List[str], max_workers: int = 5) -> Dict[str, BSEQuote]:
        out: Dict[str, BSEQuote] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_stock_quote, code): code for code in scrip_codes}
            for fut in as_completed(futures):
                quote = fut.result()
                if quote:
                    out[futures[fut]] = quote
        return out

    def get_stock_info(self, scrip_code: str) -> Optional[Dict[str, Any]]:
        if not str(scrip_code).isdigit():
            scrip_code = self.get_scrip_code(scrip_code) or scrip_code
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/ComHeader/GetQuote/Equity/{scrip_code}")
        return data if isinstance(data, dict) else None

    def get_stock_peer_comparison(self, scrip_code: str) -> Optional[pd.DataFrame]:
        if not str(scrip_code).isdigit():
            scrip_code = self.get_scrip_code(scrip_code) or scrip_code
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/EQPeerComp/{scrip_code}")
        return pd.DataFrame(data) if isinstance(data, list) else None

    def get_shareholding_pattern(self, scrip_code: str) -> Dict[str, Any]:
        if not str(scrip_code).isdigit():
            scrip_code = self.get_scrip_code(scrip_code) or scrip_code
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/CShareHoldPat/w/{scrip_code}/0")
        result = {"scrip_code": scrip_code, "quarters": []}
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    result["quarters"].append(
                        {
                            "quarter": row.get("SHPDate", ""),
                            "promoter": self._safe_float(row.get("PromotersPer", 0)),
                            "fii": self._safe_float(row.get("FIIPer", 0)),
                            "dii": self._safe_float(row.get("DIIPer", 0)),
                            "public": self._safe_float(row.get("PublicPer", 0)),
                        }
                    )
        return result

    # HISTORICAL DATA
    def get_historical_data(
        self,
        scrip_code: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "1y",
        interval: str = "daily",
    ) -> pd.DataFrame:
        if not str(scrip_code).isdigit():
            scrip_code = self.get_scrip_code(scrip_code) or scrip_code
        end_dt = self._parse_date(end_date) or datetime.now()
        start_dt = self._parse_date(start_date)
        if start_dt is None:
            start_dt = end_dt - timedelta(days={"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 365))
        data = self.session.get_json(
            f"{self.base_url}/BseIndiaAPI/api/StockPriceCSVDownload/w",
            {"scripcode": scrip_code, "FromDate": start_dt.strftime("%Y%m%d"), "ToDate": end_dt.strftime("%Y%m%d"), "owner": ""},
        )
        df = self._table_to_df(data)
        if df.empty:
            return df
        rename = {"trd_date": "date", "open_price": "open", "high_price": "high", "low_price": "low", "close_price": "close", "ttl_trd_qnty": "volume", "ttl_trd_val": "value"}
        df.rename(columns=rename, inplace=True)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if interval != "daily":
            df = self._resample_data(df, interval)
        return df

    def get_bulk_historical_data(self, scrip_codes: List[str], period: str = "1y", max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_historical_data, code, None, None, period, "daily"): code for code in scrip_codes}
            for fut in as_completed(futures):
                df = fut.result()
                if not df.empty:
                    out[futures[fut]] = df
        return out

    def get_intraday_data(self, scrip_code: str, interval: str = "5") -> pd.DataFrame:
        if not str(scrip_code).isdigit():
            scrip_code = self.get_scrip_code(scrip_code) or scrip_code
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/StockReachGraph/w", {"scripcode": scrip_code, "flag": "I", "interval": interval})
        return self._table_to_df(data)

    def _resample_data(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        if df.empty or "date" not in df.columns:
            return df
        freq = {"weekly": "W", "monthly": "M", "quarterly": "Q"}.get(interval.lower())
        if not freq:
            return df
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        return df.set_index("date").resample(freq).agg({k: v for k, v in agg.items() if k in cols}).dropna().reset_index()

    # INDEX DATA
    def get_index_data(self, index_name: str = "SENSEX") -> Optional[Dict[str, Any]]:
        code = BSE_INDEX_CODES.get(index_name.upper(), "16")
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/GetSensexData/w/{code}")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            row = data[0]
            return {"index_name": index_name, "current_value": self._safe_float(row.get("ltp", 0)), "change": self._safe_float(row.get("chg", 0)), "change_pct": self._safe_float(row.get("pchg", 0))}
        return None

    def get_index_constituents(self, index_name: str = "SENSEX") -> pd.DataFrame:
        code = BSE_INDEX_CODES.get(index_name.upper(), "16")
        return self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MktRGainerLoser/w/{code}"))

    def get_index_historical(self, index_name: str = "SENSEX", start_date: datetime = None, end_date: datetime = None, period: str = "1y") -> pd.DataFrame:
        end_dt = end_date or datetime.now()
        start_dt = start_date or (end_dt - timedelta(days={"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(period, 365)))
        code = BSE_INDEX_CODES.get(index_name.upper(), "16")
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/IndexArchData/w", {"indexcode": code, "fromdate": start_dt.strftime("%d/%m/%Y"), "todate": end_dt.strftime("%d/%m/%Y")})
        return self._table_to_df(data)

    def get_all_indices(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for name in BSE_INDEX_CODES:
            d = self.get_index_data(name)
            if d:
                rows.append(d)
        return pd.DataFrame(rows)

    # MARKET DATA
    def get_gainers_losers(self, index: str = "SENSEX", category: str = "gainers") -> pd.DataFrame:
        code = BSE_INDEX_CODES.get(index.upper(), "16")
        flag = "G" if category.lower() == "gainers" else "L"
        return self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MktRGainerLoser/w/{code}/{flag}"))

    def get_top_gainers(self, index: str = "SENSEX") -> pd.DataFrame:
        return self.get_gainers_losers(index=index, category="gainers")

    def get_top_losers(self, index: str = "SENSEX") -> pd.DataFrame:
        return self.get_gainers_losers(index=index, category="losers")

    def get_52_week_high(self) -> pd.DataFrame:
        return self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MktNew52WkHighLow/w/H"))

    def get_52_week_low(self) -> pd.DataFrame:
        return self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MktNew52WkHighLow/w/L"))

    def get_most_active_volume(self) -> pd.DataFrame:
        return self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MktMostActvVol/w"))

    def get_most_active_value(self) -> pd.DataFrame:
        return self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MktMostActvVal/w"))

    def get_market_summary(self) -> Dict[str, Any]:
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MktSummary/w")
        if not isinstance(data, dict):
            return {}
        advances = self._safe_int(data.get("Advances", 0))
        declines = self._safe_int(data.get("Declines", 0))
        return {"advances": advances, "declines": declines, "unchanged": self._safe_int(data.get("Unchanged", 0)), "advance_decline_ratio": advances / max(declines, 1)}

    def get_market_breadth(self) -> Dict[str, Any]:
        summary = self.get_market_summary()
        if not summary:
            return {}
        advances = summary["advances"]
        declines = summary["declines"]
        return {"advances": advances, "declines": declines, "breadth_signal": "BULLISH" if advances > declines else "BEARISH"}

    def get_circuit_breakers(self) -> pd.DataFrame:
        upper = self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MktCircuitBreaker/w/U"))
        lower = self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MktCircuitBreaker/w/L"))
        if not upper.empty:
            upper["circuit_type"] = "UPPER"
        if not lower.empty:
            lower["circuit_type"] = "LOWER"
        return pd.concat([upper, lower], ignore_index=True) if (not upper.empty or not lower.empty) else pd.DataFrame()

    # CORPORATE ACTIONS
    def get_corporate_actions(self, scrip_code: str = None, action_type: str = "all", from_date: datetime = None, to_date: datetime = None) -> pd.DataFrame:
        to_dt = to_date or datetime.now()
        from_dt = from_date or (to_dt - timedelta(days=365))
        purpose = {"all": "", "dividend": "Dividend", "bonus": "Bonus", "split": "Stock  Split", "rights": "Rights"}.get(action_type.lower(), "")
        if scrip_code:
            if not str(scrip_code).isdigit():
                scrip_code = self.get_scrip_code(scrip_code) or scrip_code
            data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/CorporateAction/w", {"scripcode": scrip_code, "segment": "Equity", "fromdate": from_dt.strftime("%Y%m%d"), "todate": to_dt.strftime("%Y%m%d"), "purpose": purpose})
        else:
            data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/DefaultData/w", {"Atea": "Corp", "Ession": purpose, "fromdate": from_dt.strftime("%Y%m%d"), "todate": to_dt.strftime("%Y%m%d")})
        return self._table_to_df(data)

    def get_upcoming_dividends(self, days_ahead: int = 30) -> pd.DataFrame:
        return self.get_corporate_actions(action_type="dividend", from_date=datetime.now(), to_date=datetime.now() + timedelta(days=days_ahead))

    def get_upcoming_bonus(self, days_ahead: int = 60) -> pd.DataFrame:
        return self.get_corporate_actions(action_type="bonus", from_date=datetime.now(), to_date=datetime.now() + timedelta(days=days_ahead))

    def get_upcoming_splits(self, days_ahead: int = 60) -> pd.DataFrame:
        return self.get_corporate_actions(action_type="split", from_date=datetime.now(), to_date=datetime.now() + timedelta(days=days_ahead))

    # DEALS & INSIDER
    def get_bulk_deals(self, from_date: datetime = None, to_date: datetime = None) -> pd.DataFrame:
        to_dt = to_date or datetime.now()
        from_dt = from_date or (to_dt - timedelta(days=30))
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/BulknBlockDeals/w", {"Type": "Bulk", "fromdate": from_dt.strftime("%Y%m%d"), "todate": to_dt.strftime("%Y%m%d")})
        return self._table_to_df(data)

    def get_block_deals(self, from_date: datetime = None, to_date: datetime = None) -> pd.DataFrame:
        to_dt = to_date or datetime.now()
        from_dt = from_date or (to_dt - timedelta(days=30))
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/BulknBlockDeals/w", {"Type": "Block", "fromdate": from_dt.strftime("%Y%m%d"), "todate": to_dt.strftime("%Y%m%d")})
        return self._table_to_df(data)

    def get_insider_trading(self, scrip_code: str = None, from_date: datetime = None, to_date: datetime = None) -> pd.DataFrame:
        to_dt = to_date or datetime.now()
        from_dt = from_date or (to_dt - timedelta(days=90))
        if scrip_code and not str(scrip_code).isdigit():
            scrip_code = self.get_scrip_code(scrip_code)
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/InsiderTrading/w", {"scripcode": scrip_code or "", "fromdate": from_dt.strftime("%Y%m%d"), "todate": to_dt.strftime("%Y%m%d")})
        return self._table_to_df(data)

    # DELIVERY DATA
    def get_delivery_data(self, scrip_code: str, from_date: datetime = None, to_date: datetime = None) -> pd.DataFrame:
        df = self.get_historical_data(scrip_code, start_date=from_date, end_date=to_date)
        if df.empty or "dly_perc" not in df.columns:
            return df
        out = df.copy()
        out["delivery_pct"] = out["dly_perc"].map(self._safe_float)
        return out

    # SECTOR DATA
    def get_sector_performance(self) -> pd.DataFrame:
        sectors = ["BANKEX", "IT", "HEALTHCARE", "AUTO", "FMCG", "METAL", "OIL&GAS", "POWER", "REALTY", "TELECOM", "ENERGY", "PSU"]
        rows = []
        for sec in sectors:
            val = self.get_index_data(sec)
            if val:
                rows.append({"sector": sec, "value": val.get("current_value", 0), "change_pct": val.get("change_pct", 0)})
        return pd.DataFrame(rows).sort_values("change_pct", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()

    # IPO DATA
    def get_upcoming_ipos(self) -> pd.DataFrame:
        return self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/IPOData/w/current"))

    def get_recent_ipos(self) -> pd.DataFrame:
        return self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/IPOData/w/recent"))

    # ANNOUNCEMENTS
    def get_announcements(self, scrip_code: str = None, from_date: datetime = None, to_date: datetime = None, category: str = "all") -> pd.DataFrame:
        _ = category
        to_dt = to_date or datetime.now()
        from_dt = from_date or (to_dt - timedelta(days=30))
        if scrip_code and not str(scrip_code).isdigit():
            scrip_code = self.get_scrip_code(scrip_code)
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/AnnSubCategoryGetData/w", {"scripcode": scrip_code or "", "fromdate": from_dt.strftime("%Y%m%d"), "todate": to_dt.strftime("%Y%m%d"), "subcategory": ""})
        return self._table_to_df(data)

    def get_board_meetings(self, from_date: datetime = None, to_date: datetime = None) -> pd.DataFrame:
        from_dt = from_date or (datetime.now() - timedelta(days=7))
        to_dt = to_date or (datetime.now() + timedelta(days=30))
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/BoardMeeting/w", {"fromdate": from_dt.strftime("%Y%m%d"), "todate": to_dt.strftime("%Y%m%d")})
        return self._table_to_df(data)

    def get_financial_results(self, scrip_code: str, quarterly: bool = True) -> pd.DataFrame:
        if not str(scrip_code).isdigit():
            scrip_code = self.get_scrip_code(scrip_code) or scrip_code
        period = "Q" if quarterly else "A"
        return self._table_to_df(self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/FinancialResult/w/{scrip_code}/{period}"))

    # FII/DII
    def get_fii_dii_activity(self, from_date: datetime = None, to_date: datetime = None) -> pd.DataFrame:
        to_dt = to_date or datetime.now()
        from_dt = from_date or (to_dt - timedelta(days=30))
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/FIIDIITradeActivity/w", {"fromdate": from_dt.strftime("%Y%m%d"), "todate": to_dt.strftime("%Y%m%d")})
        return self._table_to_df(data)

    # MUTUAL FUNDS
    def get_mutual_fund_nav(self, scheme_code: str) -> Optional[Dict[str, Any]]:
        data = self.session.get_json(f"{self.base_url}/BseIndiaAPI/api/MFNav/w/{scheme_code}")
        return data if isinstance(data, dict) else None

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            if isinstance(value, str):
                value = re.sub(r"[,$\\s₹]", "", value)
                if value in ("", "--", "-"):
                    return 0.0
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(float(BSEFetcher._safe_float(value)))
        except Exception:
            return 0

    @staticmethod
    def _parse_date(value: Optional[Union[str, datetime]]) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except Exception:
            return None

    @staticmethod
    def _table_to_df(data: Any) -> pd.DataFrame:
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for key in ("Table", "table", "Data", "data"):
                if isinstance(data.get(key), list):
                    return pd.DataFrame(data[key])
        return pd.DataFrame()


class BSEScanner:
    def __init__(self):
        self.fetcher = BSEFetcher()

    def scan_high_delivery_stocks(self, min_delivery_pct: float = 60.0, min_volume: int = 100000) -> pd.DataFrame:
        constituents = self.fetcher.get_index_constituents("BSE500")
        if constituents.empty:
            return pd.DataFrame()
        out = []
        for _, stock in constituents.head(100).iterrows():
            scrip_code = str(stock.get("SCRIP_CD", stock.get("scrip_code", "")))
            if not scrip_code:
                continue
            df = self.fetcher.get_delivery_data(scrip_code)
            if df.empty or "delivery_pct" not in df.columns:
                continue
            latest = df.iloc[-1]
            if latest.get("delivery_pct", 0) >= min_delivery_pct and latest.get("volume", 0) >= min_volume:
                out.append({"scrip_code": scrip_code, "delivery_pct": latest.get("delivery_pct", 0), "volume": latest.get("volume", 0)})
        return pd.DataFrame(out).sort_values("delivery_pct", ascending=False) if out else pd.DataFrame()

    def scan_bulk_deal_stocks(self, days: int = 7) -> pd.DataFrame:
        return self.fetcher.get_bulk_deals(from_date=datetime.now() - timedelta(days=days))

    def scan_circuit_stocks(self) -> Dict[str, pd.DataFrame]:
        df = self.fetcher.get_circuit_breakers()
        if df.empty:
            return {"upper_circuit": pd.DataFrame(), "lower_circuit": pd.DataFrame()}
        return {"upper_circuit": df[df["circuit_type"] == "UPPER"], "lower_circuit": df[df["circuit_type"] == "LOWER"]}

    def scan_fii_dii_activity(self, days: int = 10) -> Dict[str, Any]:
        df = self.fetcher.get_fii_dii_activity(from_date=datetime.now() - timedelta(days=days * 2))
        return {"data": df, "analysis": "fetched"} if not df.empty else {}

    def scan_sector_rotation(self) -> pd.DataFrame:
        df = self.fetcher.get_sector_performance()
        if df.empty:
            return df
        df = df.copy()
        df["signal"] = df["change_pct"].apply(lambda x: "INFLOW" if x > 1 else "OUTFLOW" if x < -1 else "NEUTRAL")
        return df

    def scan_52week_breakouts(self) -> Dict[str, Any]:
        highs = self.fetcher.get_52_week_high()
        lows = self.fetcher.get_52_week_low()
        return {"highs": highs, "lows": lows, "highs_count": len(highs), "lows_count": len(lows)}

    def scan_volume_leaders(self) -> Dict[str, pd.DataFrame]:
        return {"volume_leaders": self.fetcher.get_most_active_volume(), "value_leaders": self.fetcher.get_most_active_value()}

    def get_complete_market_scan(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensex": self.fetcher.get_index_data("SENSEX") or {},
            "market_breadth": self.fetcher.get_market_breadth(),
            "sector_performance": self.fetcher.get_sector_performance(),
            "top_gainers": self.fetcher.get_top_gainers(),
            "top_losers": self.fetcher.get_top_losers(),
            "circuit_breakers": self.scan_circuit_stocks(),
            "week_52": self.scan_52week_breakouts(),
            "volume_leaders": self.scan_volume_leaders(),
        }


def get_bse_quote(symbol: str) -> Optional[BSEQuote]:
    return BSEFetcher().get_stock_quote(symbol)


def get_bse_historical(symbol: str, period: str = "1y") -> pd.DataFrame:
    return BSEFetcher().get_historical_data(symbol, period=period)


def get_sensex_data() -> Optional[Dict[str, Any]]:
    return BSEFetcher().get_index_data("SENSEX")


def run_bse_scan() -> Dict[str, Any]:
    return BSEScanner().get_complete_market_scan()
