"""
sector_scanner.py — Smart Money Sector Flow Engine

Usage:
  python sector_scanner.py --init      # one-time 3-year bulk download
  python sector_scanner.py --update    # append latest trading day (run daily after close)
  python sector_scanner.py --tags      # refresh sector/industry tags from yfinance
"""

import argparse
import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

DATA_DIR = Path("sector_data")
OHLCV_FILE = DATA_DIR / "ohlcv.parquet"
UNIVERSE_FILE = DATA_DIR / "universe.json"
SECTOR_CACHE_FILE = DATA_DIR / "sector_cache.json"
THEME_FILE = Path("theme_groups.json")

DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

_SP500_FALLBACK = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE",
    "ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG",
    "AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","APO","AAPL","AMAT","APTV",
    "ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR",
    "BALL","BAC","BK","BBWI","BAX","BDX","BRK-B","BBY","TECH","BIIB","BLK","BX","BA","BCR",
    "BMY","AVGO","BR","BRO","BF-B","BLDR","BSX","BG","CDNS","CZR","CPT","CPB","COF","CAH",
    "KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CHTR",
    "CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH",
    "CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP",
    "COST","CTRA","CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DAY","DE","DELL","DAL",
    "DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK",
    "DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG",
    "EPAM","EQT","EFX","EQIX","EQR","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR",
    "XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F",
    "FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEV","GEHC","GEN","GNRC","GD",
    "GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE",
    "HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX",
    "IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH",
    "IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY",
    "KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN",
    "LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC",
    "MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP",
    "MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI",
    "MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS",
    "NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE",
    "ORCL","OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG",
    "PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PRU","PEG","PTC","PSA",
    "PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD",
    "RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SEE","SRE","NOW",
    "SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE",
    "SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY",
    "TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL",
    "TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO",
    "VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","WAB","WBA","WMT","DIS","WBD","WM",
    "WAT","WEC","WFC","WELL","WST","WDC","WRK","WY","WHR","WMB","WTW","GWW","WYNN","XEL",
    "XYL","YUM","ZBRA","ZBH","ZTS",
]

_NDX_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","GOOG","AVGO","COST","NFLX","AMD",
    "ADBE","QCOM","INTU","CSCO","INTC","TXN","AMAT","AMGN","HON","SBUX","BKNG","GILD",
    "ADP","MDLZ","REGN","VRTX","ISRG","PANW","LRCX","ADI","KLAC","SNPS","CDNS","MELI",
    "ASML","MU","MRNA","CRWD","ABNB","FTNT","DXCM","CEG","TTD","BIIB","IDXX","ROST",
    "ORLY","ODFL","PAYX","VRSK","FANG","EXC","NXPI","MCHP","PCAR","CTAS","FAST","CPRT",
    "MNST","GEHC","ON","KDP","DLTR","DDOG","ZS","TEAM","ANSS","ALGN","ILMN","OKTA",
    "HOOD","AFRM","SOFI","ARM","MRVL","PLTR","NOW","NET","SNOW","COIN","UBER","AEP",
]


def _fetch_sp500() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if "symbol" in cols_lower:
                col = t.columns[cols_lower.index("symbol")]
                tickers = t[col].dropna().str.replace(".", "-", regex=False).tolist()
                if len(tickers) > 400:
                    return tickers
    except Exception as e:
        print(f"  Wikipedia S&P500 fetch failed ({e}), using fallback list")
    return _SP500_FALLBACK


def _fetch_nasdaq100() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if "ticker" in cols_lower or "symbol" in cols_lower:
                col_idx = cols_lower.index("ticker") if "ticker" in cols_lower else cols_lower.index("symbol")
                col = t.columns[col_idx]
                tickers = t[col].dropna().tolist()
                if len(tickers) > 50:
                    return [str(s).replace(".", "-") for s in tickers]
    except Exception as e:
        print(f"  Wikipedia Nasdaq-100 fetch failed ({e}), using fallback list")
    return _NDX_FALLBACK


def _get_theme_tickers() -> list[str]:
    """All tickers referenced in theme_groups.json."""
    if not THEME_FILE.exists():
        return []
    themes = json.loads(THEME_FILE.read_text())
    tickers = []
    for v in themes.values():
        tickers.extend(v)
    return list(set(tickers))


def get_universe(force_refresh: bool = False) -> list[str]:
    if UNIVERSE_FILE.exists() and not force_refresh:
        data = json.loads(UNIVERSE_FILE.read_text())
        age_days = (datetime.now() - datetime.fromisoformat(data["updated"])).days
        # Also refresh if theme tickers have grown beyond what's saved
        saved = set(data["tickers"])
        theme_t = set(_get_theme_tickers())
        if age_days < 7 and theme_t.issubset(saved):
            return data["tickers"]

    print("Fetching S&P 500 + Nasdaq-100 universe...")
    sp500 = _fetch_sp500()
    ndx = _fetch_nasdaq100()
    theme_tickers = _get_theme_tickers()
    combined = sorted(set(sp500 + ndx + theme_tickers))
    UNIVERSE_FILE.write_text(json.dumps({
        "updated": datetime.now().isoformat(),
        "tickers": combined,
        "sp500_count": len(sp500),
        "ndx_count": len(ndx),
        "theme_count": len(theme_tickers),
    }, indent=2))
    print(f"Universe: {len(sp500)} S&P500 + {len(ndx)} Nasdaq-100 + {len(theme_tickers)} theme = {len(combined)} unique")
    return combined


def download_missing_tickers():
    """Download any tickers in the universe that are not yet in the parquet."""
    if not OHLCV_FILE.exists():
        print("No parquet yet — run --init first.")
        return

    universe = get_universe(force_refresh=True)
    existing_df = pd.read_parquet(OHLCV_FILE)
    close_cols = existing_df["Close"].columns.tolist() if "Close" in existing_df.columns.get_level_values(0) else []
    missing = [t for t in universe if t not in close_cols]

    if not missing:
        print("All universe tickers already in parquet.")
        return

    print(f"Downloading {len(missing)} missing tickers: {missing[:10]}{'...' if len(missing)>10 else ''}")
    start = existing_df.index[0].strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    batch_size = 50
    frames = []
    for i in range(0, len(missing), batch_size):
        batch = missing[i: i + batch_size]
        try:
            df = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False, threads=True)
            if not df.empty:
                frames.append(df)
                print(f"  Downloaded {min(i+batch_size, len(missing))}/{len(missing)}")
        except Exception as e:
            print(f"  Batch failed: {e}")
        time.sleep(0.5)

    if not frames:
        print("Nothing downloaded.")
        return

    new_data = pd.concat(frames, axis=1)
    new_data = new_data.loc[:, ~new_data.columns.duplicated()]
    new_data.index = pd.to_datetime(new_data.index)

    combined = pd.concat([existing_df, new_data], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.sort_index(inplace=True)
    combined.to_parquet(OHLCV_FILE)
    print(f"Parquet updated: now {combined.shape[1]} columns")

    # Tag new tickers
    tags = load_sector_tags()
    new_to_tag = [t for t in missing if t not in tags]
    if new_to_tag:
        print(f"Tagging {len(new_to_tag)} new tickers...")
        for ticker in new_to_tag:
            try:
                info = yf.Ticker(ticker).info
                tags[ticker] = {
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                }
            except Exception:
                tags[ticker] = {"sector": "Unknown", "industry": "Unknown"}
            time.sleep(0.05)
        cache = {"_updated": datetime.now().isoformat(), **tags}
        SECTOR_CACHE_FILE.write_text(json.dumps(cache, indent=2))
        print("Tags updated.")


# ---------------------------------------------------------------------------
# Sector / Industry Tagging
# ---------------------------------------------------------------------------

def refresh_sector_tags(tickers: list[str], force: bool = False) -> dict:
    cache = {}
    if SECTOR_CACHE_FILE.exists():
        cache = json.loads(SECTOR_CACHE_FILE.read_text())
        age_days = (datetime.now() - datetime.fromisoformat(cache.get("_updated", "2000-01-01"))).days
        if not force and age_days < 7:
            return {k: v for k, v in cache.items() if not k.startswith("_")}

    print(f"Tagging {len(tickers)} tickers with sector/industry (this takes ~5 min)...")
    tags = {}
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            tags[ticker] = {
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
            }
        except Exception:
            tags[ticker] = {"sector": "Unknown", "industry": "Unknown"}
        if (i + 1) % 50 == 0:
            print(f"  Tagged {i+1}/{len(tickers)}")
        time.sleep(0.05)

    cache = {"_updated": datetime.now().isoformat(), **tags}
    SECTOR_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    print(f"Sector tags saved for {len(tags)} tickers.")
    return tags


def load_sector_tags() -> dict:
    if not SECTOR_CACHE_FILE.exists():
        return {}
    data = json.loads(SECTOR_CACHE_FILE.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# OHLCV Download
# ---------------------------------------------------------------------------

def download_initial(years: int = 3):
    tickers = get_universe()
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)

    print(f"Bulk download: {len(tickers)} tickers from {start.date()} to {end.date()}...")
    print("This will take several minutes...")

    # Download in batches of 100 to avoid yfinance timeouts
    batch_size = 100
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {batch[0]}…{batch[-1]}")
        try:
            df = yf.download(
                batch,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                frames.append(df)
            else:
                # single ticker returned as flat df
                df.columns = pd.MultiIndex.from_product([df.columns, batch])
                frames.append(df)
        except Exception as e:
            print(f"  Batch failed: {e}")
        time.sleep(1)

    if not frames:
        print("No data downloaded.")
        return

    combined = pd.concat(frames, axis=1)
    # Remove duplicate ticker columns (same ticker may appear in multiple batches edge case)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.index = pd.to_datetime(combined.index)
    combined.to_parquet(OHLCV_FILE)
    print(f"Saved {len(combined)} rows × {combined.shape[1]} columns to {OHLCV_FILE}")
    print(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}")


def update_daily():
    if not OHLCV_FILE.exists():
        print("No existing data. Run with --init first.")
        return

    df = pd.read_parquet(OHLCV_FILE)
    last_date = df.index[-1].date()
    today = datetime.now().date()

    if last_date >= today:
        print(f"Already up to date (last: {last_date})")
        return

    # Download last 5 trading days to handle gaps/weekends
    start = last_date - timedelta(days=7)
    tickers = get_universe()
    print(f"Updating from {start} → {today} ({len(tickers)} tickers)...")

    batch_size = 100
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        try:
            new = yf.download(
                batch,
                start=start.strftime("%Y-%m-%d"),
                end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if not new.empty:
                frames.append(new)
        except Exception as e:
            print(f"  Batch {i//batch_size+1} failed: {e}")
        time.sleep(0.5)

    if not frames:
        print("No new data retrieved.")
        return

    new_data = pd.concat(frames, axis=1)
    new_data = new_data.loc[:, ~new_data.columns.duplicated()]
    new_data.index = pd.to_datetime(new_data.index)

    # Merge: keep existing rows not in new_data, then append new
    combined = pd.concat([df[df.index < new_data.index[0]], new_data])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined.to_parquet(OHLCV_FILE)
    added = len(combined) - len(df)
    print(f"Updated: +{added} rows. Now {len(combined)} rows, last date: {combined.index[-1].date()}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

PERIOD_MAP = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "3Y": 756}


def _get_close_volume(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract Close and Volume from multi-level columns."""
    close = df["Close"] if "Close" in df.columns.get_level_values(0) else pd.DataFrame()
    volume = df["Volume"] if "Volume" in df.columns.get_level_values(0) else pd.DataFrame()
    return close, volume


def compute_metrics(period: str = "1W") -> pd.DataFrame:
    """Returns a DataFrame with one row per group (sector/industry/theme) and all metrics."""
    if not OHLCV_FILE.exists():
        return pd.DataFrame()

    df = pd.read_parquet(OHLCV_FILE)
    close, volume = _get_close_volume(df)

    if close.empty:
        return pd.DataFrame()

    n = PERIOD_MAP.get(period, 5)
    themes = json.loads(THEME_FILE.read_text()) if THEME_FILE.exists() else {}
    sector_tags = load_sector_tags()

    # Build sector → tickers and industry → tickers maps
    sector_map: dict[str, list[str]] = {}
    industry_map: dict[str, list[str]] = {}
    for ticker, info in sector_tags.items():
        s = info.get("sector", "Unknown")
        ind = info.get("industry", "Unknown")
        if s != "Unknown":
            sector_map.setdefault(s, []).append(ticker)
        if ind != "Unknown":
            industry_map.setdefault(ind, []).append(ticker)

    spx_col = "^GSPC" if "^GSPC" in close.columns else (
        "SPY" if "SPY" in close.columns else None
    )

    rows = []
    all_groups = (
        [("Sector", k, v) for k, v in sorted(sector_map.items())]
        + [("Industry", k, v) for k, v in sorted(industry_map.items())]
        + [("Theme", k, v) for k, v in sorted(themes.items())]
    )

    for group_type, name, tickers in all_groups:
        valid = [t for t in tickers if t in close.columns]
        if len(valid) < 2:
            continue

        g_close = close[valid].dropna(how="all")
        if len(g_close) < n + 21:
            continue

        # Returns over period
        ret = (g_close.iloc[-1] / g_close.iloc[-n - 1] - 1) * 100
        avg_ret = float(ret.median())

        # 1D return for reference
        ret_1d = (g_close.iloc[-1] / g_close.iloc[-2] - 1) * 100
        avg_ret_1d = float(ret_1d.median())

        # Relative strength vs SPX
        rs = None
        if spx_col and spx_col in close.columns and len(close[spx_col].dropna()) > n:
            spx_ret = float((close[spx_col].iloc[-1] / close[spx_col].iloc[-n - 1] - 1) * 100)
            rs = round(avg_ret - spx_ret, 2)

        # Volume ratio (last day vs 20-day average)
        g_vol = volume[valid].dropna(how="all") if not volume.empty and valid[0] in volume.columns else pd.DataFrame()
        vol_ratio = None
        if not g_vol.empty and len(g_vol) >= 21:
            last_vol = g_vol.iloc[-1].sum()
            avg_vol = g_vol.iloc[-21:-1].mean().sum()
            vol_ratio = round(float(last_vol / avg_vol), 2) if avg_vol > 0 else None

        # Breadth: % of stocks above their 20-day SMA
        sma20 = g_close.rolling(20).mean()
        above = (g_close.iloc[-1] > sma20.iloc[-1]).sum()
        breadth = round(float(above / len(valid) * 100), 1)

        # Momentum score: weighted composite
        score_components = [avg_ret * 0.4]
        if rs is not None:
            score_components.append(rs * 0.4)
        if vol_ratio is not None:
            score_components.append((vol_ratio - 1) * 10 * 0.2)
        score = round(sum(score_components), 2)

        rows.append({
            "type": group_type,
            "name": name,
            "tickers": len(valid),
            f"ret_{period}": round(avg_ret, 2),
            "ret_1D": round(avg_ret_1d, 2),
            "rs_vs_spx": rs,
            "vol_ratio": vol_ratio,
            "breadth_pct": breadth,
            "score": score,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------

def get_highlights(metrics: pd.DataFrame) -> dict:
    if metrics.empty:
        return {}

    period_col = [c for c in metrics.columns if c.startswith("ret_") and c != "ret_1D"]
    ret_col = period_col[0] if period_col else "ret_1D"

    themes_df = metrics[metrics["type"] == "Theme"].copy()
    sectors_df = metrics[metrics["type"] == "Sector"].copy()

    def top_n(df: pd.DataFrame, col: str, n: int = 3, ascending: bool = False) -> list[dict]:
        sub = df.dropna(subset=[col]).sort_values(col, ascending=ascending)
        if not ascending:
            sub = sub.tail(n).iloc[::-1]
        else:
            sub = sub.head(n)
        return sub[["name", col, "vol_ratio", "breadth_pct"]].to_dict("records")

    # Smart money signature: volume surge + positive RS
    smart_money = themes_df[
        (themes_df["vol_ratio"].fillna(0) > 1.5) &
        (themes_df["rs_vs_spx"].fillna(-999) > 0)
    ].sort_values("vol_ratio", ascending=False)

    # Momentum building: 1W RS improving vs sector baseline
    accelerating = themes_df[
        (themes_df["rs_vs_spx"].fillna(-999) > 1) &
        (themes_df["breadth_pct"] > 55)
    ].sort_values("rs_vs_spx", ascending=False)

    # Reversal watch: negative RS but volume spiking (accumulation at lows)
    reversal = themes_df[
        (themes_df["vol_ratio"].fillna(0) > 1.3) &
        (themes_df[ret_col].fillna(0) < -2)
    ].sort_values("vol_ratio", ascending=False)

    return {
        "top_themes": top_n(themes_df, ret_col, n=5),
        "bottom_themes": top_n(themes_df, ret_col, n=3, ascending=True),
        "top_sectors": top_n(sectors_df, "rs_vs_spx", n=5),
        "volume_surge": smart_money.head(5)[["name", "vol_ratio", "rs_vs_spx"]].to_dict("records"),
        "momentum_building": accelerating.head(5)[["name", "rs_vs_spx", "breadth_pct"]].to_dict("records"),
        "reversal_watch": reversal.head(3)[["name", ret_col, "vol_ratio"]].to_dict("records"),
        "last_updated": datetime.now().isoformat(),
    }


def get_constituents(group_type: str, group_name: str, metrics: pd.DataFrame, period: str = "1W") -> pd.DataFrame:
    """Return per-ticker breakdown for a selected group."""
    if not OHLCV_FILE.exists():
        return pd.DataFrame()

    df = pd.read_parquet(OHLCV_FILE)
    close, volume = _get_close_volume(df)

    if group_type == "Theme":
        themes = json.loads(THEME_FILE.read_text())
        tickers = themes.get(group_name, [])
    else:
        tags = load_sector_tags()
        key = "sector" if group_type == "Sector" else "industry"
        tickers = [t for t, info in tags.items() if info.get(key) == group_name]

    valid = [t for t in tickers if t in close.columns]
    if not valid:
        return pd.DataFrame()

    n = PERIOD_MAP.get(period, 5)
    g_close = close[valid].dropna(how="all")

    rows = []
    for t in valid:
        if t not in g_close.columns or len(g_close[t].dropna()) < n + 5:
            continue
        price = float(g_close[t].iloc[-1])
        ret = float((g_close[t].iloc[-1] / g_close[t].iloc[-n - 1] - 1) * 100)
        ret_1d = float((g_close[t].iloc[-1] / g_close[t].iloc[-2] - 1) * 100)
        sma20 = float(g_close[t].rolling(20).mean().iloc[-1])
        vs_sma = round((price / sma20 - 1) * 100, 2) if sma20 > 0 else None

        vr = None
        if not volume.empty and t in volume.columns:
            last_v = volume[t].iloc[-1]
            avg_v = volume[t].iloc[-21:-1].mean()
            vr = round(float(last_v / avg_v), 2) if avg_v > 0 else None

        rows.append({
            "ticker": t,
            "price": round(price, 2),
            f"ret_{period}": round(ret, 2),
            "ret_1D": round(ret_1d, 2),
            "vs_20sma": vs_sma,
            "vol_ratio": vr,
        })

    return pd.DataFrame(rows).sort_values(f"ret_{period}", ascending=False)


# ---------------------------------------------------------------------------
# Trend Series (for drill-down charts)
# ---------------------------------------------------------------------------

def compute_trend_series(
    names: list[str],
    group_type: str,
    period: str = "3M",
) -> dict[str, pd.Series]:
    """
    Returns {name: normalized_series} where series is indexed to 100 at the
    start of the period. group_type: "Sector" | "Industry" | "Theme" | "Stock"
    """
    if not OHLCV_FILE.exists():
        return {}

    df = pd.read_parquet(OHLCV_FILE)
    close, _ = _get_close_volume(df)
    if close.empty:
        return {}

    n_bars = PERIOD_MAP.get(period, 63)
    close = close.iloc[-n_bars:] if len(close) > n_bars else close

    tags = load_sector_tags()
    themes = json.loads(THEME_FILE.read_text()) if THEME_FILE.exists() else {}

    result = {}
    for name in names:
        if group_type == "Stock":
            tickers = [name]
        elif group_type == "Theme":
            tickers = themes.get(name, [])
        elif group_type == "Sector":
            tickers = [t for t, info in tags.items() if info.get("sector") == name]
        elif group_type == "Industry":
            tickers = [t for t, info in tags.items() if info.get("industry") == name]
        else:
            continue

        valid = [t for t in tickers if t in close.columns]
        if not valid:
            continue

        prices = close[valid].ffill().dropna(how="all", axis=1)
        if prices.empty or len(prices) < 3:
            continue

        base = prices.iloc[0].replace(0, np.nan)
        normed = prices.div(base) * 100
        avg = normed.mean(axis=1).dropna()

        if len(avg) >= 3:
            result[name] = avg

    return result


def get_all_sectors() -> list[str]:
    tags = load_sector_tags()
    return sorted({info.get("sector", "Unknown") for info in tags.values()} - {"Unknown"})


def get_sector_industries(sector: str) -> list[str]:
    tags = load_sector_tags()
    return sorted({
        info.get("industry", "Unknown")
        for info in tags.values()
        if info.get("sector") == sector and info.get("industry", "Unknown") != "Unknown"
    })


def get_industry_stocks(industry: str) -> list[str]:
    tags = load_sector_tags()
    return [t for t, info in tags.items() if info.get("industry") == industry]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sector Flow Data Engine")
    parser.add_argument("--init", action="store_true", help="One-time 3-year bulk download")
    parser.add_argument("--update", action="store_true", help="Append latest trading day")
    parser.add_argument("--tags", action="store_true", help="Refresh sector/industry tags")
    parser.add_argument("--missing", action="store_true", help="Download tickers not yet in parquet")
    parser.add_argument("--years", type=int, default=3, help="Years of history for --init")
    args = parser.parse_args()

    if args.tags:
        universe = get_universe()
        refresh_sector_tags(universe, force=True)
    elif args.init:
        get_universe(force_refresh=True)
        universe = get_universe()
        download_initial(years=args.years)
        refresh_sector_tags(universe)
    elif args.update:
        update_daily()
    elif args.missing:
        download_missing_tickers()
    else:
        parser.print_help()
