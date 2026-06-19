import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from config.loader import AudioInputConfig, ModelConfig
from models.model_factory import ModelFactory
from models.whisper_model import WhisperModel
from models.omnilingual_model import OmnilingualModel
from models.vosk_model import VoskModel

class TestASRModels(unittest.TestCase):

    def setUp(self):
        # Generate 1 second of dummy audio at 16kHz
        self.sr = 16000
        self.dummy_audio = np.random.uniform(-1, 1, self.sr).astype(np.float32)

    def test_model_factory(self):
        whisper = ModelFactory.get_model("whisper", model_size="tiny")
        self.assertIsInstance(whisper, WhisperModel)

        hf_whisper = ModelFactory.get_model("hf_whisper", model_id_or_path="dummy")
        from models.hf_whisper_model import HFWhisperModel
        self.assertIsInstance(hf_whisper, HFWhisperModel)

        omnilingual = ModelFactory.get_model("omnilingual")
        self.assertIsInstance(omnilingual, OmnilingualModel)

        vosk = ModelFactory.get_model("vosk")
        self.assertIsInstance(vosk, VoskModel)

    def test_model_factory_from_config(self):
        """from_config() should produce the same model as get_model()."""
        cfg = ModelConfig(
            type="whisper",
            language="fa",
            audio=AudioInputConfig(sample_rate=16000),
            extras={"size": "tiny"},
        )
        model = ModelFactory.from_config(cfg)
        self.assertIsInstance(model, WhisperModel)

    @patch('whisper.load_model')
    def test_whisper_transcribe_logic(self, mock_load):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello world"}
        mock_load.return_value = mock_model
        
        model = WhisperModel(model_size="tiny")
        transcript = model.transcribe(self.dummy_audio)
        
        self.assertEqual(transcript, "hello world")
        mock_model.transcribe.assert_called_once()
        args, _ = mock_model.transcribe.call_args
        self.assertEqual(args[0].dtype, np.float32)

    @patch('transformers.AutoProcessor.from_pretrained')
    @patch('transformers.AutoModelForCTC.from_pretrained')
    @patch('torch.argmax')
    def test_omnilingual_transcribe_logic(self, mock_argmax, mock_model_load, mock_proc_load):
        # Mock transformers components
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        # Mock processor behavior — use Persian text so the ASCII-fallback
        # branch in OmnilingualModel.transcribe() is never triggered.
        mock_processor.return_value = MagicMock(input_values=MagicMock(to=lambda x: MagicMock()))
        mock_processor.batch_decode.return_value = ["متن فارسی"]
        
        # Mock model behavior
        mock_logits = MagicMock()
        mock_model.return_value = MagicMock(logits=mock_logits)
        mock_argmax.return_value = MagicMock()
        
        mock_proc_load.return_value = mock_processor
        mock_model_load.return_value = mock_model
        
        model = OmnilingualModel(model_id="facebook/mms-300m")
        # Manually set device to avoid torch.cuda check issues in some env
        model.device = "cpu"
        model.model = mock_model
        model.processor = mock_processor
        
        transcript = model.transcribe(self.dummy_audio, language="fa")
        
        self.assertEqual(transcript, "متن فارسی")
        mock_processor.assert_called()
        # Verify it handled the language mapping
        if hasattr(mock_processor, "set_target_lang"):
            mock_processor.set_target_lang.assert_called_with("fas")

    @patch('vosk.Model')
    @patch('vosk.KaldiRecognizer')
    def test_vosk_transcribe_logic(self, mock_rec, mock_model):
        # Mock Vosk components
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.FinalResult.return_value = '{"text": "vosk output"}'
        mock_rec.return_value = mock_recognizer
        
        with patch('os.path.exists', return_value=True):
            model = VoskModel(model_path="dummy/path")
            model.model = mock_model
            model.recognizer = mock_recognizer
            
            transcript = model.transcribe(self.dummy_audio)
            
            self.assertEqual(transcript, "vosk output")
            mock_recognizer.AcceptWaveform.assert_called_once()
            # Verify it converted to int16 bytes
            args, _ = mock_recognizer.AcceptWaveform.call_args
            self.assertIsInstance(args[0], bytes)

if __name__ == '__main__':
    unittest.main()
