import io
import logging
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

logger = logging.getLogger(__name__)

CAMB_API_BASE = "https://client.camb.ai/apis"


class VoiceCloneTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> list[ToolInvokeMessage]:
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            return [self.create_text_message("Error: CAMB AI API key is required")]

        voice_name = tool_parameters.get("voice_name", "").strip()
        gender = tool_parameters.get("gender", "male")
        audio_file_url = tool_parameters.get("audio_file_url", "").strip()

        if not voice_name:
            return [self.create_text_message("Error: Voice name is required")]
        if not audio_file_url:
            return [self.create_text_message("Error: Audio file URL is required")]

        try:
            # Download the audio file
            audio_response = requests.get(audio_file_url, timeout=60)
            audio_response.raise_for_status()
            audio_data = audio_response.content

            # Determine filename from URL
            filename = audio_file_url.split("/")[-1].split("?")[0]
            if not filename or "." not in filename:
                filename = "voice_sample.wav"

            # Create the custom voice
            response = requests.post(
                f"{CAMB_API_BASE}/create-custom-voice",
                headers={
                    "x-api-key": api_key,
                },
                files={
                    "file": (filename, io.BytesIO(audio_data), "audio/wav"),
                },
                data={
                    "voice_name": voice_name,
                    "gender": 1 if gender == "male" else 2,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            voice_id = data.get("voice_id") or data.get("id")
            voice_result_name = data.get("voice_name") or data.get("name") or voice_name

            result_text = f"Voice cloned successfully!\n\nVoice Name: {voice_result_name}\nVoice ID: {voice_id}\nGender: {gender}"

            if voice_id:
                result_text += f"\n\nYou can now use voice_id={voice_id} in TTS and Translated TTS tools."

            return [self.create_text_message(result_text)]

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 401:
                return [self.create_text_message("Error: Invalid CAMB AI API key")]
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            return [self.create_text_message(f"Error: Voice clone API error (HTTP {status_code}): {body or e}")]
        except Exception as e:
            logger.exception("Error in CAMB voice clone tool")
            return [self.create_text_message(f"Error: {str(e)}")]
