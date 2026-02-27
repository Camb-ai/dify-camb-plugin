import logging
import io
from typing import Any, Optional

import requests
from dify_plugin.entities.model import AIModelEntity
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
    InvokeError,
)
from dify_plugin import TTSModel

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"

# Reverse map: numeric language IDs to BCP-47 locale codes.
# The CAMB TTS API expects BCP-47 locale strings (e.g. "en-us"), not integers.
# We keep this map so that if Dify passes a numeric code we can convert it.
_NUMERIC_TO_BCP47: dict[int, str] = {
    1: "en-us", 2: "es-es", 3: "fr-fr", 4: "de-de", 5: "it-it", 6: "pt-br",
    7: "ja-jp", 8: "ko-kr", 9: "zh-cn", 10: "ar-sa", 11: "hi-in", 12: "nl-nl",
    13: "pl-pl", 14: "ru-ru", 15: "tr-tr", 16: "vi-vn", 17: "th-th", 18: "sv-se",
    19: "da-dk", 20: "fi-fi", 21: "id-id", 22: "ms-my", 23: "ro-ro", 24: "uk-ua",
    25: "el-gr", 26: "cs-cz", 27: "hu-hu", 28: "no-no", 29: "bg-bg", 30: "hr-hr",
    31: "sk-sk", 32: "ta-in", 33: "te-in", 34: "bn-in", 35: "ur-pk", 36: "sw-ke",
    37: "he-il", 38: "fil-ph",
}


def _resolve_language(language: str | int | None) -> str:
    """Convert a language identifier to the BCP-47 string expected by the CAMB API."""
    if language is None:
        return "en-us"
    if isinstance(language, int):
        return _NUMERIC_TO_BCP47.get(language, "en-us")
    try:
        numeric = int(language)
        return _NUMERIC_TO_BCP47.get(numeric, "en-us")
    except (ValueError, TypeError):
        pass
    # Already a BCP-47 string – pass through as lowercase
    return language.lower()


class CambTTSModel(TTSModel):
    """CAMB AI TTS model implementation for Dify."""

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> Any:
        """
        Invoke TTS model and return an audio data generator.

        :param model: model name (mars-flash, mars-pro, mars-instruct)
        :param tenant_id: tenant identifier
        :param credentials: provider credentials containing api_key
        :param content_text: text to synthesize
        :param voice: voice identifier (numeric voice_id as string)
        :param user: optional user identifier
        :return: generator yielding audio bytes chunks
        """
        api_key = credentials.get("api_key")
        if not api_key:
            raise CredentialsValidateFailedError("API key is required")

        if not content_text or not content_text.strip():
            raise InvokeBadRequestError("Content text cannot be empty")

        # Resolve voice_id -- Dify passes voice as a string
        try:
            voice_id = int(voice) if voice else 147320
        except (ValueError, TypeError):
            voice_id = 147320

        # Map model name to CAMB speech_model parameter
        speech_model_map = {
            "mars-flash": "mars-flash",
            "mars-pro": "mars-pro",
            "mars-instruct": "mars-instruct",
        }
        speech_model = speech_model_map.get(model, "mars-flash")

        return self._stream_tts(
            api_key=api_key,
            text=content_text,
            voice_id=voice_id,
            speech_model=speech_model,
        )

    def _stream_tts(
        self,
        api_key: str,
        text: str,
        voice_id: int,
        speech_model: str,
        language: str = "en-us",
    ) -> Any:
        """
        Stream TTS audio from the CAMB API.

        Uses the streaming endpoint to yield audio chunks as they arrive.
        """
        try:
            response = requests.post(
                f"{CAMB_API_BASE}/tts-stream",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "voice_id": voice_id,
                    "language": language,
                    "speech_model": speech_model,
                    "output_configuration": {
                        "format": "wav",
                    },
                },
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 401:
                raise CredentialsValidateFailedError("Invalid CAMB AI API key")
            raise InvokeError(f"CAMB TTS API error (HTTP {status}): {str(e)}")
        except requests.exceptions.Timeout:
            raise InvokeError("CAMB TTS API request timed out")
        except Exception as e:
            logger.exception("Error calling CAMB TTS API")
            raise InvokeError(f"CAMB TTS API error: {str(e)}")

    def validate_credentials(
        self,
        model: str,
        credentials: dict,
        user: Optional[str] = None,
    ) -> None:
        """Validate that credentials are correct by making a test API call."""
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
                raise CredentialsValidateFailedError("Invalid CAMB AI API key")
            response.raise_for_status()
        except CredentialsValidateFailedError:
            raise
        except Exception as e:
            raise CredentialsValidateFailedError(
                f"Failed to validate credentials: {str(e)}"
            )

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> Optional[AIModelEntity]:
        """Return None since we use predefined models only."""
        return None

    def get_tts_model_voices(
        self, model: str, credentials: dict, language: Optional[str] = None
    ) -> list[dict]:
        """
        Fetch available voices from the CAMB API.

        Returns a list of dicts with 'name' and 'value' keys.
        """
        api_key = credentials.get("api_key")
        if not api_key:
            return []

        try:
            response = requests.get(
                f"{CAMB_API_BASE}/list-voices",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            voices_data = response.json()

            voices = []
            if isinstance(voices_data, list):
                for v in voices_data:
                    voice_id = v.get("id") or v.get("voice_id")
                    voice_name = v.get("voice_name") or v.get("name") or str(voice_id)
                    if voice_id is not None:
                        voices.append({
                            "name": voice_name,
                            "value": str(voice_id),
                        })
            return voices
        except Exception as e:
            logger.warning(f"Failed to fetch CAMB voices: {e}")
            return []
