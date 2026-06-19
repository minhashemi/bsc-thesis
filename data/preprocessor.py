from __future__ import annotations

import numpy as np
import librosa

from config.loader import AudioInputConfig
from data.dataset_loader import AudioLoader, get_child_psrb_samples


class AudioPreprocessor:
    def __init__(
        self,
        audio_cfg: AudioInputConfig | None = None,
        apply_noise_reduction: bool = False,
    ):
        if audio_cfg is None:
            audio_cfg = AudioInputConfig()

        self.audio_cfg = audio_cfg
        self.apply_noise_reduction = apply_noise_reduction

        self.denoiser = None
        if self.apply_noise_reduction:
            try:
                import pyrnnoise
                self.denoiser = pyrnnoise.RNNoise(self.audio_cfg.sample_rate)
            except ImportError:
                pass

    @property
    def target_sr(self) -> int:
        return self.audio_cfg.sample_rate

    def process(self, audio_array: np.ndarray, orig_sr: int) -> np.ndarray:
        audio = audio_array.astype(np.float32)

        if self.audio_cfg.channels == "mono" and audio.ndim > 1:
            if audio.shape[1] < audio.shape[0]:
                audio = audio.T
            audio = librosa.to_mono(audio)

        target_sr = self.audio_cfg.sample_rate
        if orig_sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr)

        trim_db = self.audio_cfg.trim_silence_db
        audio, _ = librosa.effects.trim(audio, top_db=trim_db)

        if self.apply_noise_reduction:
            if self.denoiser is not None:
                frames = [
                    frame.flatten()
                    for _prob, frame in self.denoiser.denoise_chunk(audio, partial=True)
                ]
                if frames:
                    audio = np.concatenate(frames)
            else:
                try:
                    import noisereduce as nr
                    audio = nr.reduce_noise(y=audio, sr=self.audio_cfg.sample_rate)
                except ImportError as e:
                    raise ImportError("install noisereduce for noise reduction") from e

        if self.audio_cfg.normalize_volume:
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio = audio / max_amp

        return audio


class DataPipeline:
    def __init__(
        self,
        audio_cfg: AudioInputConfig | None = None,
        apply_noise_reduction: bool = False,
    ):
        self.preprocessor = AudioPreprocessor(
            audio_cfg=audio_cfg,
            apply_noise_reduction=apply_noise_reduction,
        )

    def stream_from_hf(self, dataset_name: str = "PartAI/PSRB", filter_child: bool = True):
        for audio_array, sr, transcript in AudioLoader.load_hf_stream(
            dataset_name=dataset_name, filter_child=filter_child
        ):
            yield self.preprocessor.process(audio_array, sr), transcript

    def from_local_file(self, filepath: str) -> tuple[np.ndarray, str]:
        audio_array, sr, transcript = AudioLoader.load_local_file(filepath)
        return self.preprocessor.process(audio_array, sr), transcript

    def from_cached_dataset(self, cache_dir: str = "data/psrb_child_dataset", split: str = "all"):
        for audio_array, sr, transcript in get_child_psrb_samples(cache_dir=cache_dir, split=split):
            yield self.preprocessor.process(audio_array, sr), transcript

    def from_mic(self, duration_seconds: int = 5) -> tuple[np.ndarray, str]:
        audio_array, sr, transcript = AudioLoader.record_live_mic(
            duration_seconds=duration_seconds
        )
        return self.preprocessor.process(audio_array, sr), transcript
