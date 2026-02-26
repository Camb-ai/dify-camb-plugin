import logging
import time
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"
MAX_POLL_ATTEMPTS = 120
POLL_INTERVAL = 2


class TranscribeTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> list[ToolInvokeMessage]:
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            return [self.create_text_message("Error: CAMB AI API key is required")]

        audio_file_url = tool_parameters.get("audio_file_url", "").strip()
        if not audio_file_url:
            return [self.create_text_message("Error: Audio file URL is required")]

        language = int(tool_parameters.get("language", 1))

        try:
            # Download the audio file
            audio_response = requests.get(audio_file_url, timeout=60)
            audio_response.raise_for_status()

            # Determine filename from URL
            filename = audio_file_url.split("/")[-1].split("?")[0] or "audio.wav"

            # Create transcription task
            response = requests.post(
                f"{CAMB_API_BASE}/transcribe",
                headers={"x-api-key": api_key},
                files={"media_file": (filename, audio_response.content, "audio/wav")},
                data={"language": language},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            task_id = data.get("task_id")

            if task_id is None:
                return [self.create_text_message(f"Error: API did not return a task_id: {data}")]

            # Poll for completion
            run_id = None
            for _ in range(MAX_POLL_ATTEMPTS):
                status_response = requests.get(
                    f"{CAMB_API_BASE}/transcribe/{task_id}",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
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
                    return [self.create_text_message(f"Error: Transcription failed: {error_msg}")]

                time.sleep(POLL_INTERVAL)
            else:
                return [self.create_text_message("Error: Transcription timed out")]

            # Fetch result
            if run_id:
                result_response = requests.get(
                    f"{CAMB_API_BASE}/transcription-result/{run_id}",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                    timeout=30,
                )
                result_response.raise_for_status()
                result_data = result_response.json()

                if isinstance(result_data, str):
                    return [self.create_text_message(result_data)]
                if isinstance(result_data, dict):
                    text = (
                        result_data.get("transcript")
                        or result_data.get("text")
                        or result_data.get("result")
                        or result_data.get("transcription")
                        or str(result_data)
                    )
                    if isinstance(text, list):
                        return [self.create_text_message(
                            " ".join(
                                seg.get("text", str(seg)) if isinstance(seg, dict) else str(seg)
                                for seg in text
                            )
                        )]
                    return [self.create_text_message(str(text))]
                return [self.create_text_message(str(result_data))]

            return [self.create_text_message("Error: No run_id returned")]

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 401:
                return [self.create_text_message("Error: Invalid CAMB AI API key")]
            return [self.create_text_message(f"Error: Transcription API error (HTTP {status_code}): {e}")]
        except Exception as e:
            logger.exception("Error in CAMB transcribe tool")
            return [self.create_text_message(f"Error: {str(e)}")]
