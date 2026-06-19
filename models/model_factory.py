from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.loader import ModelConfig


class ModelFactory:
    @staticmethod
    def from_config(model_cfg: "ModelConfig"):
        return ModelFactory.get_model(model_cfg.type, **model_cfg.extras)

    @staticmethod
    def get_model(model_type: str, **kwargs):
        model_type = model_type.lower()

        if model_type == "whisper":
            from models.whisper_model import WhisperModel
            return WhisperModel(model_size=kwargs.get("size", kwargs.get("model_size", "base")))

        if model_type == "hf_whisper":
            from models.hf_whisper_model import HFWhisperModel
            return HFWhisperModel(
                model_id_or_path=kwargs.get("model_id_or_path", "openai/whisper-base"),
                is_peft=kwargs.get("is_peft", False)
            )

        if model_type == "omnilingual":
            from models.omnilingual_model import OmnilingualModel
            return OmnilingualModel(
                model_id=kwargs.get("model_id", "facebook/mms-1b-all")
            )

        if model_type == "vosk":
            from models.vosk_model import VoskModel
            return VoskModel(
                model_path=kwargs.get("model_path"),
                language=kwargs.get("language", "fa"),
                model_size=kwargs.get("size", kwargs.get("model_size", "small")),
                sample_rate=kwargs.get("sample_rate", 16000),
            )

        if model_type == "gemini_direct":
            from models.gemini_direct_model import GeminiDirectModel
            return GeminiDirectModel(
                model_id=kwargs.get("model_id", "gemini-2.5-flash"),
                language=kwargs.get("language", "fa"),
                prompt_path=kwargs.get("prompt_path", "config/gemini_direct_prompt.txt"),
            )

        if model_type == "qwen_local":
            from models.qwen_audio_model import QwenAudioModel
            return QwenAudioModel(
                model_id=kwargs.get("model_id", "Qwen/Qwen2-Audio-7B-Instruct"),
                language=kwargs.get("language", "fa"),
            )

        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            "Supported types: whisper, omnilingual, vosk, gemini_direct."
        )
