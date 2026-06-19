import tempfile
import textwrap
import unittest
from pathlib import Path

from config.loader import (
    AudioInputConfig,
    load_config,
    list_profiles,
)


def _write(tmp_dir: Path, name: str, content: str) -> Path:
    p = tmp_dir / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


_MODELS_YAML = """\
    whisper:
      type: whisper
      size: base
      language: fa
      audio_input:
        sample_rate: 16000
        channels: mono
        format: float32
        trim_silence_db: 60
        normalize_volume: true

    vosk:
      type: vosk
      language: fa
      size: small
      model_path: null
      audio_input:
        sample_rate: 16000
        channels: mono
        format: int16
        trim_silence_db: 60
        normalize_volume: true
"""

_PROFILES_YAML = """\
    default_profile: raw

    profiles:
      raw:
        model_name: whisper
        data:
          source: cached
          cache_dir: data/psrb_child_dataset
          use_noise_reduction: false
        postprocessing:
          type: none
        run:
          max_samples: 5

      classic:
        model_name: vosk
        data:
          source: cached
          cache_dir: data/psrb_child_dataset
          use_noise_reduction: false
        postprocessing:
          type: classic
          lexicon_path: config/lexicon.json
        run:
          max_samples: 10
"""


class TestLoadConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.models_path = _write(self.tmp, "models.yaml", _MODELS_YAML)
        self.profiles_path = _write(self.tmp, "profiles.yaml", _PROFILES_YAML)

    def test_default_profile_resolved(self):
        """When no profile_name is given, default_profile is used."""
        cfg = load_config(
            models_path=self.models_path,
            profiles_path=self.profiles_path,
        )
        self.assertEqual(cfg.profile_name, "raw")

    def test_explicit_profile(self):
        """Requesting a specific profile loads the correct model."""
        cfg = load_config(
            profile_name="classic",
            models_path=self.models_path,
            profiles_path=self.profiles_path,
        )
        self.assertEqual(cfg.model.type, "vosk")
        self.assertEqual(cfg.postprocessing.type, "classic")

    def test_audio_input_parsed(self):
        """audio_input fields map correctly to AudioInputConfig."""
        cfg = load_config(
            profile_name="raw",
            models_path=self.models_path,
            profiles_path=self.profiles_path,
        )
        self.assertEqual(cfg.model.audio.sample_rate, 16000)
        self.assertEqual(cfg.model.audio.channels, "mono")
        self.assertEqual(cfg.model.audio.format, "float32")
        self.assertTrue(cfg.model.audio.normalize_volume)

    def test_vosk_int16_format(self):
        """Vosk profile carries format=int16 in its audio config."""
        cfg = load_config(
            profile_name="classic",
            models_path=self.models_path,
            profiles_path=self.profiles_path,
        )
        self.assertEqual(cfg.model.audio.format, "int16")

    def test_run_config(self):
        """run.max_samples is loaded correctly."""
        cfg = load_config(
            profile_name="raw",
            models_path=self.models_path,
            profiles_path=self.profiles_path,
        )
        self.assertEqual(cfg.run.max_samples, 5)

    def test_unknown_profile_raises(self):
        """Requesting a non-existent profile raises ValueError."""
        with self.assertRaises(ValueError):
            load_config(
                profile_name="nonexistent",
                models_path=self.models_path,
                profiles_path=self.profiles_path,
            )

    def test_missing_models_file_raises(self):
        """Missing models.yaml raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_config(
                models_path=self.tmp / "no_such_file.yaml",
                profiles_path=self.profiles_path,
            )

    def test_list_profiles(self):
        """list_profiles returns all profile names."""
        profiles = list_profiles(self.profiles_path)
        self.assertIn("raw", profiles)
        self.assertIn("classic", profiles)

    def test_cli_override_model(self):
        """CLI --model override replaces the model type."""
        import argparse
        ns = argparse.Namespace(model="omnilingual", language=None, source=None, max_samples=None)
        cfg = load_config(
            models_path=self.models_path,
            profiles_path=self.profiles_path,
            cli_overrides=ns,
        )
        self.assertEqual(cfg.model.type, "omnilingual")

    def test_audio_input_config_defaults(self):
        """AudioInputConfig defaults are Whisper-compatible."""
        a = AudioInputConfig()
        self.assertEqual(a.sample_rate, 16000)
        self.assertEqual(a.channels, "mono")
        self.assertEqual(a.format, "float32")


if __name__ == "__main__":
    unittest.main()
