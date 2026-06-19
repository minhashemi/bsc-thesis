from abc import ABC, abstractmethod
import numpy as np

class ASRModel(ABC):
    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def transcribe(self, audio_array: np.ndarray, sr: int = 16000) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
