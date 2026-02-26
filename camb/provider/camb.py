import logging
from typing import Any

import requests
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin import ModelProvider

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"


class CambProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        Validate CAMB AI provider credentials by attempting to list voices.
        If the API key is invalid the request will fail.
        """
        api_key = credentials.get("api_key")
        if not api_key:
            raise CredentialsValidateFailedError("API key is required")

        try:
            response = requests.get(
                f"{CAMB_API_BASE}/list-voices",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            if response.status_code == 401:
                raise CredentialsValidateFailedError(
                    "Invalid CAMB AI API key"
                )
            if response.status_code == 403:
                raise CredentialsValidateFailedError(
                    "CAMB AI API key does not have sufficient permissions"
                )
            response.raise_for_status()
        except CredentialsValidateFailedError:
            raise
        except Exception as e:
            logger.exception("Error validating CAMB AI credentials")
            raise CredentialsValidateFailedError(
                f"Failed to validate CAMB AI credentials: {str(e)}"
            )
