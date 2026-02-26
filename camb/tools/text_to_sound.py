import logging
import time
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"
MAX_POLL_ATTEMPTS = 120
POLL_INTERVAL = 3


class TextToSoundTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> list[ToolInvokeMessage]:
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            return [self.create_text_message("Error: CAMB AI API key is required")]

        prompt = tool_parameters.get("prompt", "").strip()
        if not prompt:
            return [self.create_text_message("Error: Prompt is required")]

        try:
            # Step 1: Create text-to-audio task
            response = requests.post(
                f"{CAMB_API_BASE}/text-to-sound",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": prompt,
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
                    f"{CAMB_API_BASE}/text-to-sound/{task_id}",
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
                    return [self.create_text_message(f"Error: Text to sound failed: {error_msg}")]

                time.sleep(POLL_INTERVAL)
            else:
                return [self.create_text_message("Error: Text to sound generation timed out")]

            if not run_id:
                return [self.create_text_message("Error: No run_id returned")]

            # Step 3: Fetch the audio result
            audio_response = requests.get(
                f"{CAMB_API_BASE}/text-to-sound-result/{run_id}",
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
            return [self.create_text_message(f"Error: Text to sound API error (HTTP {status_code}): {e}")]
        except Exception as e:
            logger.exception("Error in CAMB text to sound tool")
            return [self.create_text_message(f"Error: {str(e)}")]
