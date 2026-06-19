import io
import os
from pathlib import Path
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from google import genai
from google.genai import types

from models.base_model import ASRModel

_DEFAULT_PROMPT_PATH = Path("config/gemini_direct_prompt.txt")

def _load_api_key() -> str | None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GEMINI_API_KEY")
        except ImportError:
            pass
        except Exception as e:
            print(f"colab error: {e}")
    return api_key


class GeminiDirectModel(ASRModel):

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        language: str = "fa",
        prompt_path: Path | str = _DEFAULT_PROMPT_PATH,
    ):
        self._model_id = model_id
        self.language = language
        self.prompt_path = Path(prompt_path)
        self.client = None
        self.prompt = ""

    def load(self, **kwargs):
        print(f"loading gemini direct: {self._model_id}")
        from utils.gemini_client import GeminiClientWrapper
        self.client = GeminiClientWrapper()
        if not self.client.api_keys:
            print("no gemini key found, disabling direct mode")
            self.client = None

        if self.prompt_path.exists():
            self.prompt = self.prompt_path.read_text(encoding="utf-8").strip()
        else:
            print("direct prompt not found, using default")
            self.prompt = "Transcribe this audio clip into Persian text. Output only the transcription, nothing else."

    def transcribe(self, audio_array: np.ndarray, sr: int = 16000, language=None) -> str:
        if self.client is None:
            self.load()

        if self.client is None:
            return ""

        if len(audio_array) == 0:
            return ""

        try:
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, samplerate=sr, format="WAV", subtype="PCM_16")
            wav_bytes = wav_io.getvalue()

            contents = [
                types.Part.from_bytes(
                    data=wav_bytes,
                    mime_type="audio/wav",
                ),
                self.prompt,
            ]

            response = self.client.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=0.95,
                    max_output_tokens=1024,
                ),
            )

            if response and response.text:
                return response.text.strip()
            return ""

        except Exception as e:
            print(f"gemini direct error: {e}")
            return ""

    @property
    def model_name(self) -> str:
        return f"Gemini-Direct-{self._model_id}"
