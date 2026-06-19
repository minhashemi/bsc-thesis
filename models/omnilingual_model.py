import numpy as np
from models.base_model import ASRModel
from transformers import AutoProcessor, AutoModelForCTC
import torch

class OmnilingualModel(ASRModel):
    
    def __init__(self, model_id="facebook/mms-1b-all"):
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = "cpu"

    def load(self, **kwargs):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCTC.from_pretrained(self.model_id)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def transcribe(self, audio_array: np.ndarray, sr: int = 16000, language=None) -> str:
        if self.model is None:
            self.load()
            
        if language == "fa":
            language = "fas"
            
        if language:
            try:
                if hasattr(self.processor, "set_target_lang"):
                    self.processor.set_target_lang(language)
                elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "set_target_lang"):
                    self.processor.tokenizer.set_target_lang(language)
                
                if hasattr(self.model, "load_adapter"):
                    self.model.load_adapter(language)
                    
            except Exception as e:
                print(f"could not set target language {language}: {e}")
            
        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            logits = outputs.logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        if any(c in transcription for c in "abcdefghijklmnopqrstuvwxyz"):
             transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
             
        return transcription.strip()

    @property
    def model_name(self) -> str:
        return f"Omnilingual-{self.model_id.split('/')[-1]}"
