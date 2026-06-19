import json
import os
import zipfile
from pathlib import Path

import numpy as np
import requests
from vosk import KaldiRecognizer, Model

from models.base_model import ASRModel


class VoskModel(ASRModel):

    MODEL_URLS = {
        "fa": {
            "small": "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.5.zip",
            "large": "https://alphacephei.com/vosk/models/vosk-model-fa-0.5.zip",
        },
        "en": {
            "small": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "large": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.15.zip",
        },
    }

    def __init__(self, model_path: str = None, language: str = "fa", model_size: str = "small", sample_rate: int = 16000):
        self.language = language
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model_dir = Path("models")

        url_map = self.MODEL_URLS.get(language, self.MODEL_URLS["en"])
        self._download_url = url_map.get(model_size, url_map["small"])

        if model_path:
            self.model_path = model_path
        else:
            zip_name = self._download_url.split("/")[-1]
            self.model_path = str(self.model_dir / zip_name.replace(".zip", ""))

        self.model = None
        self.recognizer = None
        self.vocabulary = None

    def _download_model(self):
        if Path(self.model_path).exists():
            return

        zip_path = self.model_dir / self._download_url.split("/")[-1]

        print(f"downloading vosk: {self.model_size}")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        response = requests.get(self._download_url, stream=True)

        try:
            from tqdm import tqdm  # noqa: PLC0415
            total_size = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except ImportError:
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"extracting: {self.model_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.model_dir)

        os.remove(zip_path)

    def set_vocabulary(self, words: list):
        if words:
            cleaned_words = [w.replace("\u200c", "") for w in words]
            self.vocabulary = json.dumps(cleaned_words, ensure_ascii=False)
            print(f"vocab restricted: {len(words)}")

    def load(self, **kwargs):
        self._download_model()

        if not Path(self.model_path).exists():
            print(f"vosk missing: {self.model_path}")
            return

        print(f"loading vosk: {self.model_path}")
        self.model = Model(self.model_path)

        if not self.vocabulary:
            lexicon_path = Path("config/lexicon.json")
            if lexicon_path.exists():
                try:
                    with open(lexicon_path, "r", encoding="utf-8") as f:
                        lexicon_data = json.load(f)
                        if "fuzzy_vocab" in lexicon_data:
                            self.set_vocabulary(lexicon_data["fuzzy_vocab"])
                except Exception as e:
                    print(f"failed load lexicon: {e}")

        if self.vocabulary:
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate, self.vocabulary)
        else:
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

    def transcribe(self, audio_array: np.ndarray, sr: int = 16000, **kwargs) -> str:
        if self.model is None:
            self.load()

        if self.model is None:
            return ""

        audio_int16 = (audio_array * 32767).clip(-32768, 32767).astype(np.int16)
        self.recognizer.AcceptWaveform(audio_int16.tobytes())
        result = json.loads(self.recognizer.FinalResult())
        return result.get("text", "").strip()

    @property
    def model_name(self) -> str:
        return f"Vosk-{Path(self.model_path).name}"
