import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.normalization import PersianNormalizer
from evaluation.postprocessing.base_corrector import Corrector
from evaluation.postprocessing.llm_corrector import LLMCorrector, _build_system_prompt
from evaluation.postprocessing.rule_based import RuleBasedCorrector


def _write_temp_lexicon(data: dict) -> Path:
    """Write a lexicon dict to a temp JSON file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    )
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


def _write_temp_prompt(text: str) -> Path:
    """Write a prompt string to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", encoding="utf-8", delete=False
    )
    tmp.write(text)
    tmp.close()
    return Path(tmp.name)

class TestCorrectorABC(unittest.TestCase):
    """Verify the ABC is correctly enforced."""

    def test_cannot_instantiate_abstract_class(self):
        with self.assertRaises(TypeError):
            Corrector()  # type: ignore

    def test_concrete_subclass_must_implement_correct(self):
        class BadCorrector(Corrector):
            pass  # missing correct()

        with self.assertRaises(TypeError):
            BadCorrector()

    def test_valid_subclass_is_accepted(self):
        class NoopCorrector(Corrector):
            def correct(self, text: str) -> str:
                return text

        c = NoopCorrector()
        self.assertEqual(c.correct("test"), "test")


class TestRuleBasedCorrector(unittest.TestCase):

    LEXICON = {
        "exact": {
            "خاستم": "خواستم",
            "میره": "می‌رود",
        },
        "abbrevs": {
            "دکتر": "دکتر",
        },
        "fuzzy_vocab": [
            "خواستم",
            "می‌رود",
            "مدرسه",
        ],
    }

    def setUp(self):
        self.lexicon_path = _write_temp_lexicon(self.LEXICON)
        self.corrector = RuleBasedCorrector(lexicon_path=str(self.lexicon_path))

    # Exact match 

    def test_exact_match_single_word(self):
        self.assertEqual(self.corrector.correct("خاستم"), "خواستم")

    def test_exact_match_in_sentence(self):
        result = self.corrector.correct("من خاستم برم")
        self.assertIn("خواستم", result)

    def test_exact_match_unknown_word_unchanged(self):
        result = self.corrector.correct("سلام")
        self.assertEqual(result, "سلام")

    def test_abbreviation_match(self):
        self.assertEqual(self.corrector.correct("دکتر"), "دکتر")

    # Fuzzy match -------------------------------------------------------

    def test_fuzzy_match_close_word(self):
        # "مدرسه" is in vocabulary; a slight variant should be corrected
        result = self.corrector.apply_fuzzy_match("مدرسه")
        self.assertEqual(result, "مدرسه")

    def test_fuzzy_match_skips_short_words(self):
        # words shorter than min_length=4 should never be touched
        result = self.corrector.apply_fuzzy_match("من")
        self.assertEqual(result, "من")

    # Edge cases

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.corrector.correct(""), "")

    @patch("builtins.print")
    def test_missing_lexicon_file_does_not_crash(self, mock_print):
        corrector = RuleBasedCorrector(lexicon_path="/nonexistent/path/lexicon.json")
        self.assertEqual(corrector.correct("hello"), "hello")
        mock_print.assert_called()

    def test_isinstance_corrector(self):
        self.assertIsInstance(self.corrector, Corrector)


# llm corrector
class TestLLMCorrectorOffline(unittest.TestCase):
    """
    Tests that do NOT make real API calls.
    The Gemini client is mocked throughout.
    """

    LEXICON = {"exact": {}, "abbrevs": {}, "fuzzy_vocab": []}
    PROMPT_TEMPLATE = "<task>{ASR_HYPOTHESIS}</task><lexicon>{LEXICON}</lexicon>"

    def setUp(self):
        self.lexicon_path = _write_temp_lexicon(self.LEXICON)
        self.prompt_path = _write_temp_prompt(self.PROMPT_TEMPLATE)


    def test_build_prompt_injects_lexicon(self):
        prompt = _build_system_prompt(self.prompt_path)
        self.assertEqual(prompt, self.PROMPT_TEMPLATE)

    @patch("builtins.print")
    def test_build_prompt_missing_prompt_file(self, mock_print):
        prompt = _build_system_prompt(Path("/no/such/file.txt"))
        self.assertEqual(prompt, "")
        mock_print.assert_called()

    @patch("builtins.print")
    @patch("utils.gemini_client._load_api_keys", return_value=[])
    def test_no_api_key_disables_client(self, mock_load_keys, mock_print):
        corrector = LLMCorrector(
            prompt_path=self.prompt_path, lexicon_path=self.lexicon_path
        )
        self.assertIsNone(corrector.client)
        mock_print.assert_called()

    @patch("builtins.print")
    @patch("utils.gemini_client._load_api_keys", return_value=[])
    def test_correct_returns_original_when_disabled(self, mock_load_keys, mock_print):
        corrector = LLMCorrector(
            prompt_path=self.prompt_path, lexicon_path=self.lexicon_path
        )
        self.assertEqual(corrector.correct("original text"), "original text")

    @patch("utils.gemini_client._load_api_keys", return_value=["fake-key"])
    @patch("google.genai.Client")
    def test_correct_returns_api_response(self, mock_client_cls, mock_load_keys):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "  corrected text  "
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        corrector = LLMCorrector(
            prompt_path=self.prompt_path, lexicon_path=self.lexicon_path
        )
        result = corrector.correct("raw text")
        self.assertEqual(result, "corrected text")

    @patch("utils.gemini_client._load_api_keys", return_value=["fake-key"])
    @patch("google.genai.Client")
    def test_correct_strips_label_prefix(self, mock_client_cls, mock_load_keys):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Corrected Hypothesis: cleaned output"
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        corrector = LLMCorrector(
            prompt_path=self.prompt_path, lexicon_path=self.lexicon_path
        )
        result = corrector.correct("raw")
        self.assertEqual(result, "cleaned output")

    @patch("builtins.print")
    @patch("utils.gemini_client._load_api_keys", return_value=["fake-key"])
    @patch("google.genai.Client")
    def test_correct_falls_back_on_api_exception(self, mock_client_cls, mock_load_keys, mock_print):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
        mock_client_cls.return_value = mock_client

        corrector = LLMCorrector(
            prompt_path=self.prompt_path, lexicon_path=self.lexicon_path
        )
        result = corrector.correct("original")
        self.assertEqual(result, "original")
        mock_print.assert_called()

    @patch("utils.gemini_client._load_api_keys", return_value=["fake-key"])
    @patch("google.genai.Client")
    def test_correct_empty_string_skips_api(self, mock_client_cls, mock_load_keys):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        corrector = LLMCorrector(
            prompt_path=self.prompt_path, lexicon_path=self.lexicon_path
        )
        result = corrector.correct("   ")
        mock_client.models.generate_content.assert_not_called()
        self.assertEqual(result, "   ")

    @patch("utils.gemini_client._load_api_keys", return_value=["fake-key"])
    @patch("google.genai.Client")
    @patch("time.sleep")
    def test_correct_chunks_multiple_sentences_and_sleeps(self, mock_sleep, mock_client_cls, mock_load_keys):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Corrected Hypothesis: output chunk"
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        corrector = LLMCorrector(
            prompt_path=self.prompt_path, lexicon_path=self.lexicon_path, rate_limit_delay=1.5
        )

        # 3 sentences should split into 2 chunks (sentences 1 & 2 in chunk 1, sentence 3 in chunk 2)
        three_sentence_text = "جمله اول است. جمله دوم است. جمله سوم است."

        result = corrector.correct(three_sentence_text)

        # Verify it split and processed both chunks
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

        # Verify sleep was called exactly once (between the 2 chunks)
        mock_sleep.assert_called_once_with(1.5)

        # Verify results were joined back
        self.assertEqual(result, "output chunk output chunk")

    @patch("builtins.print")
    def test_isinstance_corrector(self, mock_print):
        with patch("utils.gemini_client._load_api_keys", return_value=[]):
            corrector = LLMCorrector(
                prompt_path=self.prompt_path, lexicon_path=self.lexicon_path
            )
        self.assertIsInstance(corrector, Corrector)
        mock_print.assert_called()


class TestPersianNormalizer(unittest.TestCase):

    def setUp(self):
        self.normalizer = PersianNormalizer()

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.normalizer.normalize(""), "")

    def test_strips_punctuation(self):
        result = self.normalizer.normalize("سلام!")
        # Exclamation mark should be removed
        self.assertNotIn("!", result)

    def test_collapses_extra_whitespace(self):
        result = self.normalizer.normalize("سلام   دنیا")
        self.assertNotIn("  ", result)
        self.assertEqual(result.strip(), result)

    def test_preserves_persian_letters(self):
        text = "مدرسه"
        result = self.normalizer.normalize(text)
        self.assertIn("مدرسه", result)

    def test_preserves_zwnj(self):
        # Zero-width non-joiner (U+200C) used in Persian compound words
        text = "می\u200cروم"
        result = self.normalizer.normalize(text)
        self.assertIn("\u200c", result)

    def test_mixed_persian_english_text(self):
        # hazm normalises Persian punctuation/ZWNJ but does not strip Latin chars.
        # Persian words must survive; English words pass through unchanged.
        result = self.normalizer.normalize("hello سلام")
        self.assertIn("سلام", result)
        self.assertIn("hello", result)

    def test_idempotent(self):
        text = "سلام دنیا"
        first = self.normalizer.normalize(text)
        second = self.normalizer.normalize(first)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
