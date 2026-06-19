import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from models.gemini_direct_model import GeminiDirectModel
from models.model_factory import ModelFactory


class TestGeminiDirectModel(unittest.TestCase):

    def setUp(self):
        # 1 second of dummy noise at 16kHz
        self.sr = 16000
        self.dummy_audio = np.random.uniform(-1, 1, self.sr).astype(np.float32)

    @patch("models.gemini_direct_model.genai.Client")
    @patch("models.gemini_direct_model._load_api_key")
    def test_gemini_direct_transcribe_logic(self, mock_load_api_key, mock_client_class):
        mock_load_api_key.return_value = "fake_api_key_for_testing"
        
        # Setup the mock client and its response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "سلام دنیا"
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Initialize model
        model = GeminiDirectModel(
            model_id="gemini-3.5-flash",
            prompt_path="config/gemini_direct_prompt.txt"
        )
        model.load()

        # Call transcribe
        transcript = model.transcribe(self.dummy_audio, sr=self.sr)

        # Assertions
        self.assertEqual(transcript, "سلام دنیا")
        mock_client.models.generate_content.assert_called_once()
        
        # Extract call args and verify structure
        args, kwargs = mock_client.models.generate_content.call_args
        self.assertEqual(kwargs.get("model"), "gemini-3.5-flash")
        
        contents = kwargs.get("contents")
        self.assertEqual(len(contents), 2)
        
        # First element should be a Part containing the audio WAV bytes
        audio_part = contents[0]
        self.assertEqual(audio_part.inline_data.mime_type, "audio/wav")
        self.assertIsInstance(audio_part.inline_data.data, bytes)
        self.assertTrue(len(audio_part.inline_data.data) > 0)
        
        # Second element should be the system prompt/instructions
        self.assertIn("ASR system", contents[1])

    def test_model_factory_for_gemini_direct(self):
        """ModelFactory should correctly build GeminiDirectModel from string and config."""
        model = ModelFactory.get_model("gemini_direct", model_id="gemini-3.0-flash")
        self.assertIsInstance(model, GeminiDirectModel)
        self.assertEqual(model._model_id, "gemini-3.0-flash")
        self.assertEqual(model.model_name, "Gemini-Direct-gemini-3.0-flash")


if __name__ == "__main__":
    unittest.main()
