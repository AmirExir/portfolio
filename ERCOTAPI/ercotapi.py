import os
import requests
from typing import Optional, Dict, Any


class ErcotAPI:
    """
    Simple client for interacting with ERCOT Public Reports API (https://api.ercot.com/api/public-reports).
    Requires environment variables or direct API key input.
    """

    def __init__(
            self,
            public_key: Optional[str] = None,
            bearer_token: Optional[str] = None,
            username: Optional[str] = None,
            password: Optional[str] = None,
            client_id: Optional[str] = None,
        ):
        self.public_key = public_key or os.getenv("ERCOT_PUBLIC_KEY")
        self.bearer_token = bearer_token or os.getenv("ERCOT_BEARER_TOKEN")
        username = username or os.getenv("ERCOT_USERNAME")
        password = password or os.getenv("ERCOT_PASSWORD")
        client_id = client_id or os.getenv("ERCOT_CLIENT_ID")

        if not self.public_key:
            print("⚠️ Warning: Missing ERCOT_PUBLIC_KEY. Set it as env var or pass to constructor.")

        if not self.bearer_token and username and password and client_id:
            self.get_bearer_token(username, password, client_id)

    def get_bearer_token(self, username: str, password: str, client_id: str, scope: str = "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access") -> None:
        """Retrieve a bearer token from ERCOT's OAuth endpoint and store it."""
        token_url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
        data = {
            "grant_type": "password",
            "scope": scope,
            "client_id": client_id,
            "username": username,
            "password": password,
        }
        print("🔐 Requesting bearer token from ERCOT...")
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token_info = response.json()
        self.bearer_token = token_info.get("access_token") 
        if self.bearer_token:
            print("✅ Bearer token acquired.")
        else:
            raise ValueError("Failed to obtain bearer token.")

    def _make_request(self, base_url: str, endpoint: str, key: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Internal method to send a GET request to ERCOT API."""
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        # Prepare headers
        headers = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if key:
            headers["Ocp-Apim-Subscription-Key"] = key
        
        print(f"🌐 Requesting: {url}")
        print(f"🧾 Headers: {headers}")
        print(f"🔍 Params: {params}")
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_public(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """Query ERCOT Public Reports API (base: https://api.ercot.com/api/public-reports)"""
        if not self.public_key and not self.bearer_token:
            raise ValueError("Missing ERCOT Public API key or Bearer token")

        base_url = "https://api.ercot.com/api/public-reports"
        print(f"🔄 Using Public Reports API: {base_url}/{endpoint}")

        return self._make_request(base_url, endpoint, self.public_key, params)



if __name__ == "__main__":
    api = ErcotAPI()

    try:
        print("\n--- Testing Public Reports API (Hourly Resource Outage Capacity) ---")
        params = {
            "operatingDateFrom": "2025-10-01",
            "operatingDateTo": "2025-10-02",
            "page": 1,
            "size": 3
        }
        report_data = api.get_public("np3-233-cd/hourly_res_outage_cap", params=params)
        print("✅ Public Reports API connected! Metadata:")
        if "_meta" in report_data:
            print(report_data["_meta"])
        if "data" in report_data and isinstance(report_data["data"], list):
            print("📊 Sample records:")
            for record in report_data["data"][:3]:
                print(record)
    except Exception as e:
        print("❌ Public API error:", e)