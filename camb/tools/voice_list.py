import json
import logging
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"


class VoiceListTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> list[ToolInvokeMessage]:
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            return [self.create_text_message("Error: CAMB AI API key is required")]

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

            if not voices_data:
                return [self.create_text_message("No voices found.")]

            if isinstance(voices_data, list):
                # Format voices into a readable list
                voice_entries = []
                for v in voices_data:
                    voice_id = v.get("id") or v.get("voice_id")
                    voice_name = v.get("voice_name") or v.get("name") or "Unknown"
                    gender = v.get("gender", "")
                    language = v.get("language", "")
                    accent = v.get("accent", "")

                    entry = f"- ID: {voice_id} | Name: {voice_name}"
                    if gender:
                        entry += f" | Gender: {gender}"
                    if language:
                        entry += f" | Language: {language}"
                    if accent:
                        entry += f" | Accent: {accent}"

                    voice_entries.append(entry)

                result = f"Available Voices ({len(voice_entries)} total):\n\n"
                result += "\n".join(voice_entries)
                return [self.create_text_message(result)]

            # If the response is not a list, return it as JSON
            return [self.create_text_message(json.dumps(voices_data, indent=2))]

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 401:
                return [self.create_text_message("Error: Invalid CAMB AI API key")]
            return [self.create_text_message(f"Error: Voice list API error (HTTP {status_code}): {e}")]
        except Exception as e:
            logger.exception("Error in CAMB voice list tool")
            return [self.create_text_message(f"Error: {str(e)}")]
