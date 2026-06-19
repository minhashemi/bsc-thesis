from rapidfuzz import distance
import json
from pathlib import Path

from evaluation.postprocessing.base_corrector import Corrector


class RuleBasedCorrector(Corrector):

    def __init__(self, lexicon_path: str = "config/lexicon.json"):
        self.exact: dict = {}
        self.abbrevs: dict = {}
        self.fuzzy_vocab: list = []

        path = Path(lexicon_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.exact = data.get("exact", {})
            self.abbrevs = data.get("abbrevs", {})
            self.fuzzy_vocab = data.get("fuzzy_vocab", [])
        else:
            print("lexicon not found, rule-based corrector disabled")

    def apply_exact_match(self, text: str) -> str:
        words = text.split()
        corrected = [
            self.exact.get(word, self.abbrevs.get(word, word))
            for word in words
        ]
        return " ".join(corrected)

    def apply_fuzzy_match(self, text: str, max_distance: int = 2, min_length: int = 4) -> str:
        if not self.fuzzy_vocab:
            return text

        words = text.split()
        corrected = []

        for word in words:
            if len(word) < min_length or word in self.fuzzy_vocab:
                corrected.append(word)
                continue

            best_match = None
            best_dist = float('inf')
            
            for vocab_word in self.fuzzy_vocab:
                dist = distance.Levenshtein.distance(word, vocab_word)
                if dist <= max_distance and dist < best_dist:
                    best_match = vocab_word
                    best_dist = dist
                    
            if best_match:
                corrected.append(best_match)
            else:
                corrected.append(word)

        return " ".join(corrected)

    def correct(self, text: str) -> str:
        if not text:
            return text

        text = self.apply_exact_match(text)
        text = self.apply_fuzzy_match(text)
        return text
