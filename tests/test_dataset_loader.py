import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import soundfile as sf

from data.dataset_loader import AudioLoader, get_child_psrb_samples
from data.preprocessor import DataPipeline

HF_TOKEN = os.environ.get("HF_TOKEN")
SKIP_HF = unittest.skipUnless(HF_TOKEN, "HF_TOKEN not set – skipping Hugging Face integration test")


class TestAudioLoaderHFStream(unittest.TestCase):
    """Integration tests: streams one sample from HF and validates its structure."""

    @SKIP_HF
    def test_stream_returns_tuple_of_three(self):
        stream = AudioLoader.load_hf_stream(filter_child=True)
        result = next(stream)
        self.assertEqual(len(result), 3, "Expected (audio_array, sample_rate, transcript)")

    @SKIP_HF
    def test_stream_audio_is_numpy_array(self):
        stream = AudioLoader.load_hf_stream(filter_child=True)
        audio, sr, transcript = next(stream)
        self.assertIsInstance(audio, np.ndarray)

    @SKIP_HF
    def test_stream_sample_rate_is_positive_int(self):
        stream = AudioLoader.load_hf_stream(filter_child=True)
        _, sr, _ = next(stream)
        self.assertIsInstance(sr, int)
        self.assertGreater(sr, 0)

    @SKIP_HF
    def test_stream_transcript_is_nonempty_string(self):
        stream = AudioLoader.load_hf_stream(filter_child=True)
        _, _, transcript = next(stream)
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0, "Transcript should not be empty")


class TestDataPipelineHFStream(unittest.TestCase):
    """Integration test: one sample goes through the full loader → preprocessor pipeline."""

    @SKIP_HF
    def test_pipeline_stream_produces_mono_normalized_audio(self):
        pipeline = DataPipeline(apply_noise_reduction=True)
        audio, transcript = next(pipeline.stream_from_hf(filter_child=True))

        self.assertEqual(audio.ndim, 1, "Preprocessed audio must be mono")
        self.assertLessEqual(
            np.max(np.abs(audio)), 1.0 + 1e-5,
            "Peak amplitude must be ≤ 1.0 after normalization",
        )
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0)


class TestDataPipelineSaveOutput(unittest.TestCase):
    """Integration test: processes one sample and saves it so you can listen."""

    @SKIP_HF
    def test_save_processed_sample(self):
        import os

        pipeline = DataPipeline(apply_noise_reduction=True)
        audio, transcript = next(pipeline.stream_from_hf(filter_child=True))

        out_dir = "data/test_outputs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "sample_processed.wav")
        sf.write(out_path, audio, 16000)

        print(f"\nTranscript : {transcript[:80]}...")
        print(f"Audio shape: {audio.shape}")
        print(f"Max amp    : {np.max(np.abs(audio)):.4f}")
        print(f"Saved to   : {out_path}")

        self.assertTrue(os.path.exists(out_path))


class TestSyntheticChildSamples(unittest.TestCase):
    @patch("data.dataset_loader.Path.exists")
    @patch("data.dataset_loader.Path.glob")
    @patch("soundfile.read")
    def test_get_synthetic_child_samples_mocked(self, mock_sf_read, mock_glob, mock_exists):
        # Setup mocks to simulate presence of the dataset
        mock_exists.return_value = True
        
        mock_path_aahu = MagicMock()
        mock_path_aahu.stem = "child1_aahu"
        mock_path_aahu.__str__.return_value = "data/Persian KidSpeech_Data/کلمات/child1_aahu.wav"
        
        mock_path_sib = MagicMock()
        mock_path_sib.stem = "child2_sib"
        mock_path_sib.__str__.return_value = "data/Persian KidSpeech_Data/کلمات/child2_sib.wav"
        
        mock_glob.return_value = [mock_path_aahu, mock_path_sib]
        mock_sf_read.return_value = (np.ones(16000), 16000)
        
        from data.dataset_loader import get_synthetic_child_samples
        samples = get_synthetic_child_samples(num_sentences=5, words_per_sentence=3)
        
        self.assertEqual(len(samples), 5)
        for audio, sr, transcript in samples:
            self.assertEqual(sr, 16000)
            self.assertIsInstance(audio, np.ndarray)
            # 3 words of 16000 samples each + 2 silence pads of 0.2s * 16000 = 3200 samples each
            # Total expected samples = 16000 * 3 + 3200 * 2 = 48000 + 6400 = 54400
            self.assertEqual(len(audio), 54400)
            words = transcript.split()
            self.assertEqual(len(words), 3)
            for w in words:
                self.assertIn(w, ["آهو", "سیب"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
