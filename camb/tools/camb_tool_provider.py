import logging

import requests
from dify_plugin import ToolProvider

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"


class CambToolProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict) -> None:
        api_key = credentials.get("api_key")
        if not api_key:
            raise Exception("API key is required")

        response = requests.get(
            f"{CAMB_API_BASE}/list-voices",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            timeout=30,
        )
        if response.status_code in (401, 403):
            raise Exception("Invalid or unauthorized CAMB AI API key")
        response.raise_for_status()
