import os
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig
from models.base_model import ASRModel

class HFWhisperModel(ASRModel):

    def __init__(self, model_id_or_path="openai/whisper-base", is_peft=False):
        self.model_id_or_path = model_id_or_path
        self.is_peft = is_peft
        self.processor = None
        self.model = None
        self.device = "cpu"

    def load(self, **kwargs):
        print(f"loading hf whisper: {self.model_id_or_path}")
        
        # detect PEFT adapter directory
        is_peft_dir = False
        if os.path.isdir(self.model_id_or_path):
            if os.path.exists(os.path.join(self.model_id_or_path, "adapter_config.json")):
                is_peft_dir = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.is_peft or is_peft_dir:
            print("loading PEFT adapter...")
            peft_config = PeftConfig.from_pretrained(self.model_id_or_path)
            base_model_id = peft_config.base_model_name_or_path
            
            self.processor = WhisperProcessor.from_pretrained(base_model_id)
            base_model = WhisperForConditionalGeneration.from_pretrained(
                base_model_id,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_id_or_path)
        else:
            self.processor = WhisperProcessor.from_pretrained(self.model_id_or_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id_or_path,
                device_map="auto" if torch.cuda.is_available() else None
            )

        if not torch.cuda.is_available():
            self.model.to(self.device)

    def transcribe(self, audio_array: np.ndarray, sr: int = 16000, language=None) -> str:
        if self.model is None:
            self.load()

        if language == "fa":
            language = "persian"

        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt")
        
        # dtype matching for PEFT/FP16 models
        model_dtype = next(self.model.parameters()).dtype
        if model_dtype == torch.uint8:
            model_dtype = torch.float16
            
        input_features = inputs.input_features.to(self.device, dtype=model_dtype)

        forced_decoder_ids = None
        prompt_ids = None
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
            
            prompt_path = "config/whisper_prompt.txt"
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                if hasattr(self.processor, "get_prompt_ids"):
                    prompt_ids = self.processor.get_prompt_ids(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generate_kwargs = {
                "forced_decoder_ids": forced_decoder_ids,
            }
            if prompt_ids is not None:
                generate_kwargs["prompt_ids"] = prompt_ids
                
            predicted_ids = self.model.generate(
                input_features, 
                **generate_kwargs
            )

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    @property
    def model_name(self) -> str:
        return f"HFWhisper-{os.path.basename(self.model_id_or_path)}"
