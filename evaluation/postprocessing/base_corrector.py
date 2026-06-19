from abc import ABC, abstractmethod


class Corrector(ABC):
    @abstractmethod
    def correct(self, text: str) -> str:
        pass
