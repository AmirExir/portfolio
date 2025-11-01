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
        base_url_public: str = "https://api.ercot.com/api/public/v1",
        base_url_esr: str = "https://api.ercot.com/api/esr/v1"
    ):
        self.public_key = public_key or os.getenv("ERCOT_PUBLIC_KEY")
        self.esr_key = esr_key or os.getenv("ERCOT_ESR_KEY")
        self.base_url_public = base_url_public
        self.base_url_esr = base_url_esr

        if not self.public_key:
            print("⚠️ Warning: Missing ERCOT_PUBLIC_KEY. Set it as env var or pass to constructor.")
        if not self.esr_key:
            print("⚠️ Warning: Missing ERCOT_ESR_KEY. Set it as env var or pass to constructor.")

    def _make_request(self, base_url: str, endpoint: str, key: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Internal method to send a GET request to ERCOT API."""
        url = f"{base_url}/{endpoint.lstrip('/')}"
        headers = {"Ocp-Apim-Subscription-Key": key}

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_public(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Query ERCOT Public API"""
        if not self.public_key:
            raise ValueError("Missing ERCOT Public API key")
        return self._make_request(self.base_url_public, endpoint, self.public_key, params)

    def get_esr(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Query ERCOT ESR API"""
        if not self.esr_key:
            raise ValueError("Missing ERCOT ESR API key")
        return self._make_request(self.base_url_esr, endpoint, self.esr_key, params)



if __name__ == "__main__":
    api = ErcotAPI()  

    try:
        system_data = api.get_public("systemconditions")
        print("✅ Public API connected! Example data:")
        print(system_data.get("items", system_data)[:2])
    except Exception as e:
        print("❌ Public API connection error:", e)

    try:
        esr_data = api.get_esr("resources")
        print("✅ ESR API connected! Example data:")
        print(esr_data.get("items", esr_data)[:2])
    except Exception as e:
        print("❌ ESR API connection error:", e)