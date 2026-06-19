import os
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

if os.environ.get("HF_TOKEN") is None:
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except (ImportError, Exception):
        pass

try:
    import sounddevice as sd
except OSError:
    sd = None


def _fix_psrb_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "audio_duration": "text",
        "number_of_speakers": "audio_duration",
        "text": "number_of_speakers",
    })


class AudioLoader:
    """
    loads audio from HF, local or mic
    """

    @staticmethod
    def load_hf_stream(dataset_name: str = "PartAI/PSRB", split: str = "train", filter_child: bool = True):
        """
        stream to download only relative records

        yield:
            (audio_array: np.ndarray, sample_rate: int, transcript: str)
        """
        print("Streaming dataset from HF...")
        token = os.environ.get("HF_TOKEN")

        labels_path = hf_hub_download(
            repo_id=dataset_name, filename="Labels.csv", repo_type="dataset", token=token
        )
        df = _fix_psrb_columns(pd.read_csv(labels_path))

        if filter_child:
            df = df[df["age"] == "child"]

        for _, row in df.iterrows():
            try:
                local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=row["audio_path"],
                    repo_type="dataset",
                    token=token,
                )
                audio_array, sr = sf.read(local_path)
                yield (audio_array, sr, row["text"])
            except Exception as e:
                print(f"Error loading {row['audio_path']}: {e}")

    @staticmethod
    def load_local_file(filepath: str):
        """
        load local audio

        return:
            (audio_array: np.ndarray, sample_rate: int, transcript: str)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Loading {filepath}")
        audio_array, sr = sf.read(str(path))
        return audio_array, sr, f"<local_file: {path.name}>"

    @staticmethod
    def record_live_mic(duration_seconds: int = 5, sr: int = 16000):
        """
        live audio (default mic)

        return:
            (audio_array: np.ndarray, sample_rate: int, transcript: str)
        """
        if sd is None:
            raise ImportError(
                "mic not found. live audio is unavailable in this env."
            )

        print("Recording from mic...")
        recording = sd.rec(int(duration_seconds * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        print("Done.")
        return recording.flatten(), sr, "<live_recording>"


def get_child_psrb_samples(cache_dir: str = "data/psrb_child_dataset", split: str = "all", test_size: float = 0.1, random_state: int = 42):
    """
    download and cache psrb samples

    arg:
        cache_dir: Path to the pickle cache file.
        split: 'all', 'train', or 'test'
        test_size: Proportion of the dataset to use for testing
        random_state: Random seed for deterministic splitting

    return:
        [(audio_array: np.ndarray, sample_rate: int, transcript: str)] 
    """
    cache_path = Path(cache_dir)
    samples = []

    if cache_path.exists():
        print("Loading from cache...")
        with open(cache_path, "rb") as f:
            samples = pickle.load(f)
    else:
        print("Downloading from HF...")
        token = os.environ.get("HF_TOKEN")

        labels_path = hf_hub_download(
            repo_id="PartAI/PSRB", filename="Labels.csv", repo_type="dataset", token=token
        )
        df = _fix_psrb_columns(pd.read_csv(labels_path))
        child_df = df[df["age"] == "child"]

        print("Downloading files...")
        for _, row in child_df.iterrows():
            try:
                local_path = hf_hub_download(
                    repo_id="PartAI/PSRB",
                    filename=row["audio_path"],
                    repo_type="dataset",
                    token=token,
                )
                audio_array, sr = sf.read(local_path)
                samples.append((audio_array, sr, row["text"]))
            except Exception as e:
                print(f"Error downloading {row['audio_path']}: {e}")

        print("Saving cache...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)

    if split != "all":
        import random
        rng = random.Random(random_state)
        shuffled_samples = samples.copy()
        rng.shuffle(shuffled_samples)
        
        n_samples = len(shuffled_samples)
        train_end = int(n_samples * 0.8)
        val_end = train_end + int(n_samples * 0.1)
        
        if split == "train":
            samples = shuffled_samples[:train_end]
            print(f"Train split: {len(samples)}")
        elif split == "val":
            samples = shuffled_samples[train_end:val_end]
            print(f"Val split: {len(samples)}")
        elif split == "test":
            samples = shuffled_samples[val_end:]
            print(f"Test split: {len(samples)}")
        else:
            raise ValueError(f"Unknown split: {split}")

    return samples


def get_synthetic_child_samples(num_sentences: int = 100, words_per_sentence: int = 6, random_state: int = 42) -> list[tuple[np.ndarray, int, str]]:
    import random
    import re
    import subprocess
    import soundfile as sf
    import librosa
    
    rar_path = Path("data/Persian_KidSpeech_Data.rar")
    extract_dir = Path("data/Persian KidSpeech_Data")
    
    if not extract_dir.exists():
        if not rar_path.exists():
            print("Downloading dataset...")
            import urllib.request
            url = "https://github.com/DSP-UT/Persian-Kids-Speech-Data-Set/raw/main/Persian_KidSpeech_Data.rar"
            # Ensure parent directories exist
            rar_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, rar_path)
            
        print("Extracting dataset...")
        try:
            subprocess.run(["unar", "-o", "data", str(rar_path)], stdout=subprocess.DEVNULL, check=True)
        except (FileNotFoundError, subprocess.SubprocessError, Exception):
            try:
                # Fall back to unrar if unar is not present
                subprocess.run(["unrar", "x", str(rar_path), "data/"], stdout=subprocess.DEVNULL, check=True)
            except (FileNotFoundError, subprocess.SubprocessError, Exception):
                print("warning: install 'unar' to extract rar dataset")
                return []
        
    words_dir = extract_dir / "کلمات"
    if not words_dir.exists():
        print("warning: words directory not found")
        return []
        
    # Get all wav files in words_dir
    wav_files = list(words_dir.glob("*.wav"))
    if not wav_files:
        return []
        
    # Mapping table
    pinglish_to_persian = {
        "aahu": "آهو", "ghataar": "قطار", "gheychi": "قیچی", "gusfand": "گوسفند",
        "havaapeymaa": "هواپیما", "havij": "هویج", "juje": "جوجه", "kaase": "کاسه",
        "kafsh": "کفش", "kafshduz": "کفشدوز", "kafshduzak": "کفشدوزک", "asb": "اسب",
        "ketaab": "کتاب", "khaane": "خانه", "laakposht": "لاک‌پشت", "maah": "ماه",
        "maahi": "ماهی", "mesvaak": "مسواک", "ney": "نی", "paa": "پا",
        "parvaane": "پروانه", "piyaaz": "پیاز", "baadkonak": "بادکنک", "pul": "پول",
        "saaat": "ساعت", "sib": "سیب", "telefon": "تلفن", "tup": "توپ",
        "yek": "یک", "barg": "برگ", "chatr": "چتر", "cheshm": "چشم",
        "dampaei": "دمپایی", "dam+paa+i": "دمپایی", "derakht": "درخت",
        "dokhtar": "دختر", "gaav": "گاو"
    }
    
    # Parse files and get word audio list
    parsed_words = []
    for f in wav_files:
        name_part = f.stem.split("_")[-1].lower()
        # Remove numbers
        name_clean = re.sub(r'\d+', '', name_part)
        persian_word = pinglish_to_persian.get(name_clean)
        if persian_word:
            parsed_words.append((f, persian_word))
            
    if not parsed_words:
        return []
        
    rng = random.Random(random_state)
    synthetic_samples = []
    
    print(f"Stitching {num_sentences} sentences...")
    for _ in range(num_sentences):
        # Choose random words to stitch
        chosen = rng.choices(parsed_words, k=words_per_sentence)
        
        audio_segments = []
        transcripts = []
        
        for f, word in chosen:
            # Read and resample to 16000Hz
            try:
                audio_array, sr = sf.read(str(f))
                if sr != 16000:
                    audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=16000)
                audio_segments.append(audio_array)
                transcripts.append(word)
            except Exception as e:
                pass
                
        if audio_segments:
            # Concatenate audio segments with a small silence (0.2s of silence) in between
            silence_samples = int(16000 * 0.2)
            silence = np.zeros(silence_samples)
            
            stitched_audio = audio_segments[0]
            for seg in audio_segments[1:]:
                stitched_audio = np.concatenate([stitched_audio, silence, seg])
                
            stitched_transcript = " ".join(transcripts)
            synthetic_samples.append((stitched_audio, 16000, stitched_transcript))
            
    return synthetic_samples