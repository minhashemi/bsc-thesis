import numpy as np
import torch
from models.base_model import ASRModel

class QwenAudioModel(ASRModel):

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-Audio-7B-Instruct",
        language: str = "fa",
    ):
        self._model_id = model_id
        self.language = language
        self.processor = None
        self.model = None

    def load(self, **kwargs):
        print(f"loading qwen local: {self._model_id}")
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self._model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self._model_id, 
            device_map="auto",
            torch_dtype=torch.float16
        )

    def transcribe(self, audio_array: np.ndarray, sr: int = 16000, language=None) -> str:
        if self.processor is None or self.model is None:
            self.load()

        if len(audio_array) == 0:
            return ""

        try:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "dummy.wav"},
                    {"type": "text", "text": "Transcribe this audio into Persian text. Output only the transcription, nothing else."}
                ]}
            ]
            
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, audio=[audio_array], sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device)

            generate_ids = self.model.generate(**inputs, max_new_tokens=256)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            
            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()

        except Exception as e:
            print(f"qwen transcribe error: {e}")
            return ""

    @property
    def model_name(self) -> str:
        return f"Qwen2-Audio-Local"
