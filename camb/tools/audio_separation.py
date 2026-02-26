import io
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


class AudioSeparationTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> list[ToolInvokeMessage]:
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            return [self.create_text_message("Error: CAMB AI API key is required")]

        audio_file_url = tool_parameters.get("audio_file_url", "").strip()
        if not audio_file_url:
            return [self.create_text_message("Error: Audio file URL is required")]

        try:
            # Download the audio file
            download_response = requests.get(audio_file_url, timeout=120)
            download_response.raise_for_status()
            audio_data = download_response.content

            # Determine filename from URL
            filename = audio_file_url.split("/")[-1].split("?")[0]
            if not filename or "." not in filename:
                filename = "audio.wav"

            # Step 1: Create audio separation task
            response = requests.post(
                f"{CAMB_API_BASE}/audio-separation",
                headers={
                    "x-api-key": api_key,
                },
                files={
                    "media_file": (filename, io.BytesIO(audio_data), "audio/wav"),
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
                    f"{CAMB_API_BASE}/audio-separation/{task_id}",
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
                    return [self.create_text_message(f"Error: Audio separation failed: {error_msg}")]

                time.sleep(POLL_INTERVAL)
            else:
                return [self.create_text_message("Error: Audio separation timed out")]

            if not run_id:
                return [self.create_text_message("Error: No run_id returned")]

            # Step 3: Fetch separation result info
            result_response = requests.get(
                f"{CAMB_API_BASE}/audio-separation-result/{run_id}",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            result_response.raise_for_status()
            result_data = result_response.json()

            # Extract foreground and background URLs
            if isinstance(result_data, dict):
                foreground_url = result_data.get("foreground_audio_url")
                background_url = result_data.get("background_audio_url")

                result_text = "Audio separation completed!\n\n"
                if foreground_url:
                    result_text += f"Vocals (Foreground): {foreground_url}\n"
                if background_url:
                    result_text += f"Instrumental (Background): {background_url}\n"

                if not foreground_url and not background_url:
                    result_text += f"Result data: {result_data}"

                return [self.create_text_message(result_text)]

            return [self.create_text_message(f"Audio separation result: {result_data}")]

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 401:
                return [self.create_text_message("Error: Invalid CAMB AI API key")]
            return [self.create_text_message(f"Error: Audio separation API error (HTTP {status_code}): {e}")]
        except Exception as e:
            logger.exception("Error in CAMB audio separation tool")
            return [self.create_text_message(f"Error: {str(e)}")]
