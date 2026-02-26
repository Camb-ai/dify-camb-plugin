import logging
import time
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"
MAX_POLL_ATTEMPTS = 180
POLL_INTERVAL = 3


class TranslatedTTSTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> list[ToolInvokeMessage]:
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            return [self.create_text_message("Error: CAMB AI API key is required")]

        text = tool_parameters.get("text", "").strip()
        source_language = int(tool_parameters.get("source_language", 1))
        target_language = int(tool_parameters.get("target_language", 2))
        voice_id = int(tool_parameters.get("voice_id", 147320))

        if not text:
            return [self.create_text_message("Error: Text is required")]

        try:
            # Step 1: Create translated TTS task
            response = requests.post(
                f"{CAMB_API_BASE}/translated-tts",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                    "voice_id": voice_id,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            task_id = data.get("task_id")

            if task_id is None:
                return [self.create_text_message(f"Error: API did not return a task_id: {data}")]

            # Step 2: Poll for completion
            run_id = None
            for _ in range(MAX_POLL_ATTEMPTS):
                status_response = requests.get(
                    f"{CAMB_API_BASE}/translated-tts/{task_id}",
                    headers={
                        "x-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                status = status_data.get("status", "").upper()

                if status in ("SUCCESS", "COMPLETED"):
                    run_id = status_data.get("run_id")
                    break

                if status in ("FAILED", "ERROR"):
                    error_msg = status_data.get("error", "Unknown error")
                    return [self.create_text_message(f"Error: Translated TTS failed: {error_msg}")]

                time.sleep(POLL_INTERVAL)
            else:
                return [self.create_text_message("Error: Translated TTS timed out")]

            if not run_id:
                return [self.create_text_message("Error: No run_id returned")]

            # Step 3: Fetch the audio result
            audio_response = requests.get(
                f"{CAMB_API_BASE}/tts-result/{run_id}",
                headers={
                    "x-api-key": api_key,
                },
                timeout=120,
            )
            audio_response.raise_for_status()

            audio_data = audio_response.content
            if not audio_data:
                return [self.create_text_message("Error: No audio data received")]

            return [
                self.create_blob_message(
                    blob=audio_data,
                    meta={
                        "mime_type": "audio/wav",
                    },
                )
            ]

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 401:
                return [self.create_text_message("Error: Invalid CAMB AI API key")]
            return [self.create_text_message(f"Error: Translated TTS API error (HTTP {status_code}): {e}")]
        except Exception as e:
            logger.exception("Error in CAMB translated TTS tool")
            return [self.create_text_message(f"Error: {str(e)}")]
