from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_MODELS_PATH = Path("config/models.yaml")
_DEFAULT_PROFILES_PATH = Path("config/profiles.yaml")

@dataclass
class AudioInputConfig:
    sample_rate: int = 16000
    channels: str = "mono"
    format: str = "float32"
    trim_silence_db: float = 60.0
    normalize_volume: bool = True


@dataclass
class ModelConfig:
    type: str = "whisper"
    language: str = "fa"
    audio: AudioInputConfig = field(default_factory=AudioInputConfig)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    source: str = "cached"
    cache_dir: str = "data/psrb_child_dataset"
    use_noise_reduction: bool = False
    use_vocab_boosting: bool = False


@dataclass
class PostprocessingConfig:
    type: str = "none"
    lexicon_path: str = "config/lexicon.json"
    llm_model: str = "gemini-2.0-flash"
    provider: str = "google"
    prompt_path: str = "config/llm_prompt.txt"
    rate_limit_delay: float = 2.0


@dataclass
class RunConfig:
    max_samples: int = 3
    exclude_samples: list[int] = field(default_factory=list)


@dataclass
class PipelineConfig:
    profile_name: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_audio_input(raw: dict) -> AudioInputConfig:
    return AudioInputConfig(
        sample_rate=raw.get("sample_rate", 16000),
        channels=raw.get("channels", "mono"),
        format=raw.get("format", "float32"),
        trim_silence_db=float(raw.get("trim_silence_db", 60.0)),
        normalize_volume=bool(raw.get("normalize_volume", True)),
    )


def _parse_model(model_name: str, models_raw: dict, profile_model_overrides: dict) -> ModelConfig:
    base = models_raw.get(model_name, {})
    merged = {**base, **profile_model_overrides}

    audio_raw = merged.pop("audio_input", {})
    audio = _parse_audio_input(audio_raw)

    model_type = merged.pop("type", "whisper")
    language = merged.pop("language", "fa")

    extras = {k: v for k, v in merged.items() if v is not None}

    return ModelConfig(type=model_type, language=language, audio=audio, extras=extras)


def _parse_data(raw: dict) -> DataConfig:
    return DataConfig(
        source=raw.get("source", "cached"),
        cache_dir=raw.get("cache_dir", "data/psrb_child_dataset"),
        use_noise_reduction=raw.get("use_noise_reduction", False),
        use_vocab_boosting=raw.get("use_vocab_boosting", False),
    )


def _parse_postprocessing(raw: dict) -> PostprocessingConfig:
    return PostprocessingConfig(
        type=raw.get("type", "none"),
        lexicon_path=raw.get("lexicon_path", "config/lexicon.json"),
        llm_model=raw.get("llm_model", "gemini-2.0-flash"),
        provider=raw.get("provider", "google"),
        prompt_path=raw.get("prompt_path", "config/llm_prompt.txt"),
        rate_limit_delay=float(raw.get("rate_limit_delay", 2.0)),
    )


def _parse_run(raw: dict) -> RunConfig:
    exclude = raw.get("exclude_samples", [])
    if isinstance(exclude, int):
        exclude = [exclude]
    return RunConfig(
        max_samples=raw.get("max_samples", 3),
        exclude_samples=[int(x) for x in exclude],
    )


def load_config(
    profile_name: str | None = None,
    models_path: Path | str = _DEFAULT_MODELS_PATH,
    profiles_path: Path | str = _DEFAULT_PROFILES_PATH,
    cli_overrides: argparse.Namespace | None = None,
) -> PipelineConfig:
    models_raw = _load_yaml(Path(models_path))
    profiles_raw = _load_yaml(Path(profiles_path))

    if profile_name is None:
        profile_name = profiles_raw.get("default_profile", "llm_pipeline")

    profiles = profiles_raw.get("profiles", {})
    if profile_name not in profiles:
        available = list(profiles.keys())
        raise ValueError(f"profile not found: {profile_name}")

    profile = profiles[profile_name]

    model_name = profile.get("model_name", "whisper")
    profile_model_overrides = profile.get("model", {})
    model_cfg = _parse_model(model_name, models_raw, profile_model_overrides)

    data_cfg = _parse_data(profile.get("data", {}))
    post_cfg = _parse_postprocessing(profile.get("postprocessing", {}))
    run_cfg = _parse_run(profile.get("run", {}))

    if cli_overrides is not None:
        if getattr(cli_overrides, "model", None):
            model_cfg.type = cli_overrides.model
        if getattr(cli_overrides, "language", None):
            model_cfg.language = cli_overrides.language
        if getattr(cli_overrides, "source", None):
            data_cfg.source = cli_overrides.source
        if getattr(cli_overrides, "max_samples", None) is not None:
            run_cfg.max_samples = cli_overrides.max_samples
        if getattr(cli_overrides, "exclude_samples", None):
            try:
                run_cfg.exclude_samples = [int(x.strip()) for x in cli_overrides.exclude_samples.split(",")]
            except ValueError:
                print("invalid format for --exclude-samples, expect integers")

    return PipelineConfig(
        profile_name=profile_name,
        model=model_cfg,
        data=data_cfg,
        postprocessing=post_cfg,
        run=run_cfg,
    )

def list_profiles(profiles_path: Path | str = _DEFAULT_PROFILES_PATH) -> list[str]:
    profiles_raw = _load_yaml(Path(profiles_path))
    return list(profiles_raw.get("profiles", {}).keys())
