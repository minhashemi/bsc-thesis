import re

try:
    from hazm import Normalizer
except ImportError:
    Normalizer = None


class PersianNormalizer:

    def __init__(self):
        if Normalizer:
            self.normalizer = Normalizer()
        else:
            self.normalizer = None
            print("hazm missing, skipping normalization")

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        if self.normalizer:
            text = self.normalizer.normalize(text)

        text = re.sub(r"[^\w\s\u200c]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text
