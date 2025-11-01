import os
import requests
from typing import Optional, Dict, Any


class ErcotAPI:
    """
    Simple client for interacting with ERCOT Public and ESR APIs.
    Requires environment variables or direct API key input.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        esr_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
        base_url_public: str = "https://api.ercot.com/api/public/v1",
        base_url_esr: str = "https://api.ercot.com/api/esr/v1"
    ):
        self.public_key = public_key or os.getenv("ERCOT_PUBLIC_KEY")
        self.esr_key = esr_key or os.getenv("ERCOT_ESR_KEY")
        self.bearer_token = bearer_token or os.getenv("ERCOT_BEARER_TOKEN")
        self.base_url_public = base_url_public
        self.base_url_esr = base_url_esr

        if not self.public_key:
            print("⚠️ Warning: Missing ERCOT_PUBLIC_KEY. Set it as env var or pass to constructor.")
        if not self.esr_key:
            print("⚠️ Warning: Missing ERCOT_ESR_KEY. Set it as env var or pass to constructor.")

    def _make_request(self, base_url: str, endpoint: str, key: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Internal method to send a GET request to ERCOT API."""
        url = f"{base_url}/{endpoint.lstrip('/')}"
        if self.bearer_token and "public-reports" in base_url:
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
        else:
            headers = {"Ocp-Apim-Subscription-Key": key}

        # Future implementation for Bearer token authentication:
        # if key.startswith("Bearer "):
        #     headers = {"Authorization": key}

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_public(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Query ERCOT Public API, Public Reports, or Public Data API automatically"""
        if not self.public_key and not self.bearer_token:
            raise ValueError("Missing ERCOT Public API key or Bearer token")

        # Detect which base URL to use
        if endpoint.lower().startswith("np"):
            base_url = "https://api.ercot.com/api/public-reports"
            print(f"🔄 Using Public Reports API: {base_url}/{endpoint}")
        elif "public-data" in endpoint.lower():
            base_url = "https://api.ercot.com"
            print(f"🔄 Using Public Data API: {base_url}/{endpoint}")
        else:
            base_url = self.base_url_public
            print(f"🔄 Using Public API: {base_url}/{endpoint}")

        return self._make_request(base_url, endpoint, self.public_key, params)

    def get_esr(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Query ERCOT ESR API"""
        if not self.esr_key:
            raise ValueError("Missing ERCOT ESR API key")
        return self._make_request(self.base_url_esr, endpoint, self.esr_key, params)



if __name__ == "__main__":
    api = ErcotAPI()

    try:
        print("\n--- Testing Public Data API ---")
        products = api.get_public("api/public-data/")
        print("✅ Connected! Listing first few data products:")
        if isinstance(products, list) and len(products) > 0:
            for p in products[:5]:
                print(f"📄 {p.get('name')} ({p.get('productId')})")
        else:
            print(products)

        print("\n--- Testing Public Reports API (np...) ---")
        params = {"deliveryDateFrom": "2025-10-01", "deliveryDateTo": "2025-10-02", "page": 1, "size": 3}
        report_data = api.get_public("np3-911-er/2d_agg_as_offers_ecrsm", params=params)
        print("✅ Public Reports API connected! Metadata:")
        if "_meta" in report_data:
            print(report_data["_meta"])
        if "data" in report_data and isinstance(report_data["data"], list):
            print("📊 Sample records:")
            for record in report_data["data"][:3]:
                print(record)
    except Exception as e:
        print("❌ Public API error:", e)

    try:
        print("\n--- Testing ESR API ---")
        esr_data = api.get_esr("storage-resources")
        print("✅ ESR API connected! Storage resources data:")
        print(esr_data)
    except Exception as e:
        print("❌ ESR API connection error:", e)