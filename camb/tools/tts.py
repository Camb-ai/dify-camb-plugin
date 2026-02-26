import logging
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"


class TTSTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> list[ToolInvokeMessage]:
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            return [self.create_text_message("Error: CAMB AI API key is required")]

        text = tool_parameters.get("text", "").strip()
        if not text:
            return [self.create_text_message("Error: Text is required")]

        voice_id = int(tool_parameters.get("voice_id", 147320))
        language = tool_parameters.get("language", "en-us")
        speech_model = tool_parameters.get("speech_model", "mars-flash")

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

            audio_data = b""
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    audio_data += chunk

            return [
                self.create_blob_message(
                    blob=audio_data,
                    meta={"mime_type": "audio/wav"},
                )
            ]

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 401:
                return [self.create_text_message("Error: Invalid CAMB AI API key")]
            return [self.create_text_message(f"Error: TTS API error (HTTP {status_code}): {e}")]
        except Exception as e:
            logger.exception("Error in CAMB TTS tool")
            return [self.create_text_message(f"Error: {str(e)}")]
