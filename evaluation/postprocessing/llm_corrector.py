import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from evaluation.postprocessing.base_corrector import Corrector

_DEFAULT_PROMPT_PATH = Path("config/llm_prompt.txt")
_DEFAULT_LEXICON_PATH = Path("config/lexicon.json")


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


def _build_system_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        print("prompt file missing, using empty prompt")
        return ""
    return prompt_path.read_text(encoding="utf-8").strip()


def get_edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return get_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class LLMCorrector(Corrector):

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        prompt_path: Path = _DEFAULT_PROMPT_PATH,
        lexicon_path: Path = _DEFAULT_LEXICON_PATH,
        rate_limit_delay: float = 2.0,
    ):
        from utils.gemini_client import GeminiClientWrapper
        self.client = GeminiClientWrapper()

        if not self.client.api_keys:
            print("no gemini key found, disabling LLM corrector")
            self.client = None
            return

        self.model_name = model_name
        self.temperature = temperature
        self.system_template = _build_system_prompt(Path(prompt_path))
        
        # Load lexicon as dictionary
        if Path(lexicon_path).exists():
            with open(lexicon_path, "r", encoding="utf-8") as f:
                try:
                    self.lexicon = json.load(f)
                except Exception:
                    self.lexicon = {"exact": {}, "abbrevs": {}, "fuzzy_vocab": []}
        else:
            self.lexicon = {"exact": {}, "abbrevs": {}, "fuzzy_vocab": []}
            
        self.rate_limit_delay = rate_limit_delay

    def correct(self, text: str) -> str:
        if not self.client or not text.strip():
            return text

        try:
            import hazm
            tokenizer = hazm.SentenceTokenizer()
            sentences = tokenizer.tokenize(text)
        except ImportError:
            import re
            sentences = [s.strip() for s in re.split(r'[.!?؟]+', text) if s.strip()]
            if not sentences:
                sentences = [text]

        chunks = []
        for i in range(0, len(sentences), 2):
            chunks.append(" ".join(sentences[i:i+2]))

        corrected_chunks = []
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            # Dynamic Context Injection (RAG-style prompt)
            import re
            chunk_words = [w.strip() for w in re.split(r'\s+', chunk) if w.strip()]
            
            dynamic_exact = {}
            dynamic_fuzzy = set()
            
            exact_dict = self.lexicon.get("exact", {})
            fuzzy_vocab = self.lexicon.get("fuzzy_vocab", [])
            
            for word in chunk_words:
                word_clean = re.sub(r'[.!?؟،,]', '', word)
                if not word_clean:
                    continue
                
                # Check for exact matches and low distance keys
                for key, val in exact_dict.items():
                    dist = get_edit_distance(word_clean, key)
                    max_len = max(len(word_clean), len(key))
                    if dist <= 1 or (max_len > 0 and dist / max_len <= 0.4):
                        dynamic_exact[key] = val
                
                # Check for fuzzy vocabulary candidates
                vocab_matches = []
                for item in fuzzy_vocab:
                    dist = get_edit_distance(word_clean, item)
                    max_len = max(len(word_clean), len(item))
                    if dist <= 1 or (max_len > 0 and dist / max_len <= 0.4):
                        vocab_matches.append((item, dist))
                
                # Take top 3 closest items for this word
                vocab_matches.sort(key=lambda x: x[1])
                for item, _ in vocab_matches[:3]:
                    dynamic_fuzzy.add(item)
            
            # Ensure literal word-keys that are present are added
            for word in chunk_words:
                word_clean = re.sub(r'[.!?؟،,]', '', word)
                if word_clean in exact_dict:
                    dynamic_exact[word_clean] = exact_dict[word_clean]
                    
            dynamic_lexicon = {
                "exact": dynamic_exact,
                "abbrevs": self.lexicon.get("abbrevs", {}),
                "fuzzy_vocab": list(dynamic_fuzzy)
            }
            dynamic_lexicon_str = json.dumps(dynamic_lexicon, ensure_ascii=False, indent=4)
            
            # Inject dynamic lexicon into system template
            system_instruction = self.system_template.replace("{LEXICON}", dynamic_lexicon_str)
            system_instruction = system_instruction.replace("{ASR_HYPOTHESIS}", chunk)
            prompt_content = f"Raw ASR Hypothesis: {chunk}"

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        top_p=0.95,
                        top_k=64,
                        max_output_tokens=1024,
                    ),
                )

                if response and response.text:
                    corrected = response.text.strip()
                    if "Corrected Hypothesis:" in corrected:
                        corrected = corrected.split("Corrected Hypothesis:")[-1].strip()
                    corrected_chunks.append(corrected)
                else:
                    corrected_chunks.append(chunk)

            except Exception as e:
                print(f"llm error: {e}")
                corrected_chunks.append(chunk)

            if idx < len(chunks) - 1 and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        return " ".join(corrected_chunks)

