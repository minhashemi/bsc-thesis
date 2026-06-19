import os
from dotenv import load_dotenv

def _load_api_keys() -> list[str]:
    load_dotenv()
    api_key_str = os.environ.get("GEMINI_API_KEY")
    if not api_key_str:
        try:
            from google.colab import userdata
            api_key_str = userdata.get("GEMINI_API_KEY")
        except ImportError:
            pass
        except Exception as e:
            print(f"colab secret error: {e}")
    
    if not api_key_str:
        return []
    
    # parse multiple keys
    keys = []
    for k in api_key_str.replace(";", ",").split(","):
        k_clean = k.strip()
        if k_clean:
            keys.append(k_clean)
    return keys

class GeminiClientWrapper:
    # client with key rotation
    def __init__(self):
        self.api_keys = _load_api_keys()
        self.current_index = 0
        self.raw_client = None
        self.models = None
        self._init_current_client()

    def _init_current_client(self):
        if not self.api_keys:
            self.raw_client = None
            self.models = None
            return
        
        key = self.api_keys[self.current_index]
        from google import genai
        self.raw_client = genai.Client(api_key=key)
        self.models = GeminiModelsWrapper(self)

    def rotate_key(self) -> bool:
        if len(self.api_keys) <= 1:
            return False
        
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        print(f"gemini key exhausted, rotating to index {self.current_index}")
        self._init_current_client()
        return True

class GeminiModelsWrapper:
    def __init__(self, parent: GeminiClientWrapper):
        self.parent = parent

    def generate_content(self, model: str, contents, config=None, **kwargs):
        num_keys = max(1, len(self.parent.api_keys))
        
        # retry on quota limit
        for attempt in range(num_keys):
            if not self.parent.raw_client:
                raise ValueError("no API client init, key missing")
            
            try:
                return self.parent.raw_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                    **kwargs
                )
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(term in error_str for term in ["resource_exhausted", "quota", "429", "limit exceeded"])
                
                if is_rate_limit and attempt < num_keys - 1:
                    if self.parent.rotate_key():
                        continue
                
                raise e
