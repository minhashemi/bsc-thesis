import os
import whisper
import numpy as np
from models.base_model import ASRModel

class WhisperModel(ASRModel):
    
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None

    def load(self, **kwargs):
        print(f"loading whisper: {self.model_size}")
        self.model = whisper.load_model(self.model_size)

    def transcribe(self, audio_array: np.ndarray, sr: int = 16000, language=None) -> str:
        if self.model is None:
            self.load()
            
        audio = audio_array.astype(np.float32)
        
        options = {
            "fp16": False,
            "language": language,
            "condition_on_previous_text": False,
        }
        
        if language in ["fa", "persian"]:
            prompt_path = "config/whisper_prompt.txt"
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    options["initial_prompt"] = f.read().strip()
            else:
                options["initial_prompt"] = "متن زیر مربوط به صدای کودکان به زبان فارسی است:"
            
        result = self.model.transcribe(audio, **options)
        return result["text"].strip()

    @property
    def model_name(self) -> str:
        return f"Whisper-{self.model_size}"
