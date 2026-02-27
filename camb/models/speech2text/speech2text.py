import logging
import time
from typing import IO, Optional

import requests
from dify_plugin import Speech2TextModel
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
    InvokeError,
)

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"

# Maximum number of polling attempts before giving up
MAX_POLL_ATTEMPTS = 120
# Seconds to wait between polling requests
POLL_INTERVAL = 2


class CambSpeech2TextModel(Speech2TextModel):
    """CAMB AI Speech-to-Text model implementation for Dify."""

    def _invoke(
        self,
        model: str,
        credentials: dict,
        file: IO[bytes],
        user: Optional[str] = None,
    ) -> str:
        """
        Invoke the CAMB transcription API.

        :param model: model name (camb-transcribe)
        :param credentials: provider credentials containing api_key
        :param file: audio file to transcribe
        :param user: optional user identifier
        :return: transcribed text string
        """
        api_key = credentials.get("api_key")
        if not api_key:
            raise CredentialsValidateFailedError("API key is required")

        if file is None:
            raise InvokeBadRequestError("Audio file is required for transcription")

        # Step 1: Create transcription task
        task_id = self._create_transcription(api_key, file)

        # Step 2: Poll for completion
        run_id = self._poll_transcription(api_key, task_id)

        # Step 3: Fetch the result
        transcript = self._get_transcription_result(api_key, run_id)

        return transcript

    def _create_transcription(self, api_key: str, file: IO[bytes]) -> int:
        """Submit a transcription task to CAMB API."""
        try:
            # Determine the filename
            filename = getattr(file, "name", "audio.wav")
            if "/" in filename:
                filename = filename.split("/")[-1]

            response = requests.post(
                f"{CAMB_API_BASE}/transcribe",
                headers={
                    "x-api-key": api_key,
                },
                files={
                    "media_file": (filename, file, "audio/wav"),
                },
                data={
                    "language": 1,  # Default to English; auto-detect if supported
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            task_id = data.get("task_id")
            if task_id is None:
                raise InvokeError(
                    f"CAMB API did not return a task_id: {data}"
                )
            return task_id

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 401:
                raise CredentialsValidateFailedError("Invalid CAMB AI API key")
            raise InvokeError(f"Failed to create transcription (HTTP {status}): {e}")
        except (CredentialsValidateFailedError, InvokeError):
            raise
        except Exception as e:
            logger.exception("Error creating CAMB transcription task")
            raise InvokeError(f"Failed to create transcription: {e}")

    def _poll_transcription(self, api_key: str, task_id: int) -> int:
        """Poll the transcription task until it completes. Returns the run_id."""
        for attempt in range(MAX_POLL_ATTEMPTS):
            try:
                response = requests.get(
                    f"{CAMB_API_BASE}/transcribe/{task_id}",
                    headers={
                        "x-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                status = data.get("status", "").upper()

                if status == "SUCCESS" or status == "COMPLETED":
                    run_id = data.get("run_id")
                    if run_id is None:
                        raise InvokeError(
                            f"Transcription completed but no run_id returned: {data}"
                        )
                    return run_id

                if status in ("FAILED", "ERROR"):
                    error_msg = data.get("error", "Unknown error")
                    raise InvokeError(
                        f"Transcription task failed: {error_msg}"
                    )

                # Still processing -- wait and retry
                time.sleep(POLL_INTERVAL)

            except (InvokeError, CredentialsValidateFailedError):
                raise
            except Exception as e:
                logger.warning(f"Error polling transcription status (attempt {attempt}): {e}")
                time.sleep(POLL_INTERVAL)

        raise InvokeError(
            f"Transcription timed out after {MAX_POLL_ATTEMPTS * POLL_INTERVAL} seconds"
        )

    def _get_transcription_result(self, api_key: str, run_id: int) -> str:
        """Fetch the final transcription text."""
        try:
            response = requests.get(
                f"{CAMB_API_BASE}/transcription-result/{run_id}",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # The result may be nested; try common patterns
            if isinstance(data, str):
                return data
            if isinstance(data, dict):
                text = (
                    data.get("transcript")
                    or data.get("text")
                    or data.get("result")
                    or data.get("transcription")
                    or ""
                )
                if isinstance(text, list):
                    # Could be a list of segments
                    return " ".join(
                        seg.get("text", str(seg)) if isinstance(seg, dict) else str(seg)
                        for seg in text
                    )
                return str(text)

            return str(data)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            raise InvokeError(
                f"Failed to get transcription result (HTTP {status}): {e}"
            )
        except InvokeError:
            raise
        except Exception as e:
            logger.exception("Error fetching CAMB transcription result")
            raise InvokeError(f"Failed to get transcription result: {e}")

    def validate_credentials(
        self,
        model: str,
        credentials: dict,
        user: Optional[str] = None,
    ) -> None:
        """Validate credentials by calling the list_voices endpoint."""
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
