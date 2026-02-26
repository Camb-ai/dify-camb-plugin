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


class TranslateTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> list[ToolInvokeMessage]:
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            return [self.create_text_message("Error: CAMB AI API key is required")]

        text = tool_parameters.get("text", "").strip()
        source_language = int(tool_parameters.get("source_language", 1))
        target_language = int(tool_parameters.get("target_language", 2))

        if not text:
            return [self.create_text_message("Error: Text to translate is required")]

        try:
            # Create translation task
            response = requests.post(
                f"{CAMB_API_BASE}/translate",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "texts": [text],
                    "source_language": source_language,
                    "target_language": target_language,
                },
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
                    f"{CAMB_API_BASE}/translate/{task_id}",
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
                    # Some endpoints return the translation directly
                    translated_text = status_data.get("translation") or status_data.get("translated_text")
                    if translated_text:
                        return [self.create_text_message(translated_text)]
                    break

                if status in ("FAILED", "ERROR"):
                    error_msg = status_data.get("error", "Unknown error")
                    return [self.create_text_message(f"Error: Translation failed: {error_msg}")]

                time.sleep(POLL_INTERVAL)
            else:
                return [self.create_text_message("Error: Translation timed out")]

            # Fetch translation result
            if run_id:
                result_response = requests.get(
                    f"{CAMB_API_BASE}/translation-result/{run_id}",
                    headers={
                        "x-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )
                result_response.raise_for_status()
                result_data = result_response.json()

                if isinstance(result_data, str):
                    return [self.create_text_message(result_data)]
                if isinstance(result_data, dict):
                    translated = (
                        result_data.get("translation")
                        or result_data.get("translated_text")
                        or result_data.get("text")
                        or result_data.get("result")
                        or str(result_data)
                    )
                    return [self.create_text_message(translated)]
                return [self.create_text_message(str(result_data))]

            return [self.create_text_message("Error: No run_id returned for translation")]

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 401:
                return [self.create_text_message("Error: Invalid CAMB AI API key")]
            return [self.create_text_message(f"Error: Translation API error (HTTP {status_code}): {e}")]
        except Exception as e:
            logger.exception("Error in CAMB translate tool")
            return [self.create_text_message(f"Error: {str(e)}")]
