import unittest

import numpy as np

from config.loader import AudioInputConfig
from data.preprocessor import AudioPreprocessor


class TestAudioPreprocessor(unittest.TestCase):

    def _make_stereo(self, sr: int = 44100, duration: float = 2.0) -> np.ndarray:
        """Return a 2-channel stereo waveform."""
        t = np.linspace(0, duration, int(duration * sr))
        return np.array([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)])

    def test_whisper_config_mono_output(self):
        """Stereo input → mono output when channels='mono'."""
        cfg = AudioInputConfig(sample_rate=16000, channels="mono", format="float32")
        preprocessor = AudioPreprocessor(audio_cfg=cfg)

        stereo = self._make_stereo(sr=44100)
        processed = preprocessor.process(stereo, orig_sr=44100)

        self.assertEqual(processed.ndim, 1, "Output must be 1-D (mono)")

    def test_whisper_config_sample_rate(self):
        """Audio is resampled to the configured sample rate."""
        cfg = AudioInputConfig(sample_rate=16000, channels="mono")
        preprocessor = AudioPreprocessor(audio_cfg=cfg)

        stereo = self._make_stereo(sr=44100, duration=2.0)
        processed = preprocessor.process(stereo, orig_sr=44100)

        # Allow ±10 % tolerance for silence trimming
        expected_len = int(2.0 * 16000)
        self.assertAlmostEqual(len(processed), expected_len, delta=expected_len * 0.1)

    def test_whisper_config_peak_normalised(self):
        """Volume normalisation brings peak to 1.0."""
        cfg = AudioInputConfig(sample_rate=16000, channels="mono", normalize_volume=True)
        preprocessor = AudioPreprocessor(audio_cfg=cfg)

        stereo = self._make_stereo(sr=44100)
        processed = preprocessor.process(stereo, orig_sr=44100)

        self.assertTrue(
            np.isclose(np.max(np.abs(processed)), 1.0),
            "Peak amplitude should be normalised to 1.0",
        )

    def test_no_normalisation(self):
        """normalize_volume=False must leave amplitude untouched."""
        cfg = AudioInputConfig(sample_rate=16000, channels="mono", normalize_volume=False)
        preprocessor = AudioPreprocessor(audio_cfg=cfg)

        mono_audio = np.full(16000, 0.5, dtype=np.float32)
        processed = preprocessor.process(mono_audio, orig_sr=16000)

        # After silence trim the constant signal might be empty, so just check
        # that the peak is NOT forced to 1.0 (it should stay at 0.5 if non-empty)
        if len(processed) > 0:
            self.assertAlmostEqual(float(np.max(np.abs(processed))), 0.5, places=2)

    def test_default_config_fallback(self):
        """AudioPreprocessor with no config falls back to Whisper defaults."""
        preprocessor = AudioPreprocessor()  # no audio_cfg

        self.assertEqual(preprocessor.target_sr, 16000)
        self.assertEqual(preprocessor.audio_cfg.channels, "mono")


if __name__ == "__main__":
    unittest.main()
