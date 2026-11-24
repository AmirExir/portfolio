"""Issuer-reported outstanding ETF shares for the dashboard's USD listings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from io import BytesIO
import logging
import math
import os
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from lxml import etree, html
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
import requests

from market_agent.vanguard_etf_shares import download_vanguard_etf_shares


_LOG = logging.getLogger(__name__)
_SPDR = frozenset(("SPY", "GLD", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY"))
_ISHARES = {
    "SLV": "239855/ishares-silver-trust-fund", "IWM": "239710/ishares-russell-2000-etf",
    "TLT": "239454/ishares-20-year-treasury-bond-etf",
}
ETF_LISTINGS = _SPDR.union(_ISHARES, ("QQQ", "USO", "VOO"))


def _get(url: str, *, params: dict[str, str] | None = None, user_agent: str = "Market-Agent/1.0") -> requests.Response:
    options = {"params": params, "headers": {"User-Agent": user_agent}, "timeout": (3, 8)}
    try:
        response = requests.get(url, **options)
    except (requests.Timeout, requests.ConnectionError):
        response = None
    # Retry a failed connection or gateway once. Respect explicit rate limits and backoff.
    if response is None or (response.status_code in (502, 503, 504) and "Retry-After" not in response.headers):
        response = requests.get(url, **options)
    response.raise_for_status()
    return response


def _spdr_shares(ticker: str) -> tuple[float, str, str]:
    base = os.getenv("SPDR_NAV_BASE_URL", "https://www.ssga.com/library-content/products/fund-data/etfs/us")
    response = _get(f"{base.rstrip('/')}/navhist-us-en-{ticker.lower()}.xlsx")
    workbook = load_workbook(BytesIO(response.content), read_only=True, data_only=True)
    try:
        rows = list(workbook.worksheets[0].iter_rows(max_row=5, max_col=4, values_only=True))
    finally:
        workbook.close()
    # The issuer publishes one workbook per listing, with daily rows newest first.
    if rows[1][0] != "Ticker Symbol:" or str(rows[1][1]).rstrip("®") != ticker:
        raise ValueError("Wrong ETF listing in NAV workbook")
    if rows[3][:3] != ("Date", "NAV", "Shares Outstanding"):
        raise ValueError("Missing shares-outstanding column in NAV workbook")
    stamp = rows[4][0]
    if isinstance(stamp, datetime):
        stamp = stamp.date()
    elif not isinstance(stamp, date):
        stamp = datetime.strptime(stamp, "%d-%b-%Y").date()
    return rows[4][2], "State Street", stamp.isoformat()


def _ishares_shares(ticker: str) -> tuple[float, str, str]:
    base = os.getenv("ISHARES_US_PRODUCT_URL", "https://www.ishares.com/us/products")
    doc = html.fromstring(_get(f"{base.rstrip('/')}/{_ISHARES[ticker]}").content)

    def field(key: str) -> str:
        values = {" ".join(node.text_content().split()) for node in doc.xpath('//*[@data-id=$key]', key=key)}
        if len(values) != 1:
            raise ValueError(f"Missing or conflicting iShares field {key}")
        return values.pop()

    if field("fundHeader-externalTicker") != ticker:
        raise ValueError("Wrong iShares listing")
    shares = float(field("keyFundFacts-sharesOutstanding-data").replace(",", ""))
    stamp = datetime.strptime(field("keyFundFacts-sharesOutstanding-asOf"), "as of %b %d, %Y")
    return shares, "iShares", stamp.date().isoformat()


def _invesco_shares(ticker: str) -> tuple[float, str, str]:
    base = os.getenv("INVESCO_SHARECLASS_URL", "https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses")
    data = _get(f"{base.rstrip('/')}/{ticker}", params={
        "idType": "ticker", "variationType": "fundDetails", "productType": "ETF",
    }).json()
    if not isinstance(data, dict) or (data.get("ticker"), data.get("currencyCode")) != (ticker, "USD"):
        raise ValueError("Wrong Invesco share class")
    return data["sharesOutstanding"], "Invesco", data["effectiveDate"]


def _uso_shares(today: date) -> tuple[float, str, str]:
    base = os.getenv("SEC_XBRL_CONCEPT_URL", "https://data.sec.gov/api/xbrl/companyconcept")
    concept = "EntityCommonStockSharesOutstanding"
    data = _get(
        f"{base.rstrip('/')}/CIK0001327068/dei/{concept}.json",
        user_agent=os.getenv("SEC_USER_AGENT", "Market Agent/1.0 contact@amirexirpe.com"),
    ).json()
    if not isinstance(data, dict) or int(data["cik"]) != 1327068 or (data["taxonomy"], data["tag"]) != ("dei", concept):
        raise ValueError("Wrong SEC filing concept or registrant")
    # The cover-page observation is dated separately from the quarter end and filing date.
    records = (
        record for record in data["units"]["shares"]
        if record["form"] in ("10-Q", "10-K", "10-Q/A", "10-K/A")
        and date.fromisoformat(record["end"]) <= date.fromisoformat(record["filed"]) <= today
    )
    latest = max(records, key=lambda record: (record["end"], record["filed"]))
    return latest["val"], "USCF (SEC filing)", latest["end"]


def download_etf_shares(tickers: tuple[str, ...], *, today: date | None = None) -> dict[str, tuple[float, str, str]]:
    """Look up outstanding shares, source and effective date per requested ETF listing.

    Accept daily observations up to eight calendar days old. VOO is published
    monthly and USO's filing is quarterly, with respective limits of 62 and 120 days.
    A missing, stale or invalid listing is excluded so the chart can report it.
    """
    requested = tuple(dict.fromkeys(ticker for ticker in tickers if ticker in ETF_LISTINGS))
    if not requested:
        return {}
    checked = today or datetime.now(timezone.utc).date()

    def fetch(ticker: str) -> tuple[float, str, str] | None:
        try:
            if ticker in _SPDR:
                observation = _spdr_shares(ticker)
            elif ticker in _ISHARES:
                observation = _ishares_shares(ticker)
            elif ticker == "VOO":
                observation = download_vanguard_etf_shares(ticker)
            elif ticker == "USO":
                observation = _uso_shares(checked)
            else:
                observation = _invesco_shares(ticker)
            if observation is None:
                return None
            shares, source, stamp = observation
            effective = date.fromisoformat(stamp)
            max_age = {"USO": 120, "VOO": 62}.get(ticker, 8)
            if not 0 <= (checked - effective).days <= max_age:
                raise ValueError(f"Out-of-window share count dated {effective}")
            if type(shares) not in (float, int) or not math.isfinite(shares) or shares <= 0:
                raise ValueError("Shares outstanding must be a positive finite number")
            return float(shares), source, effective.isoformat()
        except (requests.RequestException, BadZipFile, InvalidFileException, etree.LxmlError, ParseError,
                ValueError, TypeError, KeyError, IndexError, OverflowError, OSError) as exc:
            http_response = getattr(exc, "response", None)
            reason = f"HTTP {http_response.status_code}" if http_response is not None else type(exc).__name__
            _LOG.warning("ETF shares unavailable for %s (%s)", ticker, reason)
            return None

    with ThreadPoolExecutor(max_workers=min(4, len(requested))) as pool:
        return {ticker: result for ticker, result in zip(requested, pool.map(fetch, requested)) if result is not None}
