# Enhancing Automatic Speech Recognition for Child Speech in Low-Resource Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minhashemi/bsc-thesis/blob/main/bsc_thesis.ipynb)

## Project Structure

```
bsc-thesis/
├── config/                  
│   ├── models.yaml              # model and input requirements
│   ├── profiles.yaml            # exec profiles (raw, classic, llm, direct)
│   ├── loader.py                # config loader
│   ├── lexicon.json             # dict for rule-based correction
│   ├── gemini_direct_prompt.txt # direct (voice-to-LLM) audio ASR
│   └── llm_prompt.txt           # LLM post-processing
│
├── data/                        
│   ├── dataset_loader.py        # PSRB dataset loader
│   ├── preprocessor.py          # audio preprocessing
│   └── psrb_child_dataset       # cached dataset
│
├── models/                   
│   ├── base_model.py            # base class
│   ├── model_factory.py         
│   ├── gemini_direct_model.py   # audio-to-LLM  
│   ├── whisper_model.py         # Whisper 
│   ├── omnilingual_model.py     # Meta Omnilingual
│   └── vosk_model.py            # Vosk
│
├── evaluation/                  
│   ├── metrics.py               # WER, CER, Levenshtein distance
│   ├── normalization.py         # Persian normalizer
│   └── postprocessing/          
│       ├── base_corrector.py    
│       ├── rule_based.py        # rule-based corrector
│       └── llm_corrector.py     # LLM corrector
│
├── tests/                    
├── main.py                      
├── bsc_thesis.ipynb          
├── requirements.txt          
└── README.md                 
```

**Run Tests**:
   ```bash
   PYTHONPATH=. pytest tests/
   ```
