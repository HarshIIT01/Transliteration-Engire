# Transliteration Engine

A lightweight, phonetically-aware neural transliteration system that converts **Roman (Latin) script** into **Devanagari (Hindi)** script while preserving pronunciation.

```
Input:  namaste doston, mera naam harshit hai
Output: नमस्ते दोस्तों, मेरा नाम हर्षित है
```

> **Transliteration ≠ Translation**
> - Translation: *hello* → *नमस्ते* (meaning changes)
> - Transliteration: *hello* → *हेलो* (script changes, pronunciation preserved)

---

## Architecture

```
Roman Input
    │
    ▼
┌──────────────────────┐
│  Language Detection   │───► English words bypass (wifi, laptop)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Roman Normalization  │───► aa→a, ee→i, ph→f
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Dictionary Backoff   │───► O(1) lookup for top 50K words
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  BPE Tokenization     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  BiGRU Seq2Seq Model  │───► Encoder → Attention → Decoder
│  (~4M params, ~15MB)  │
└──────────┬───────────┘
           ▼
     Devanagari Output
```

## Model Details

| Component | Specification |
|-----------|--------------|
| Encoder | Bidirectional GRU (256 hidden) |
| Embedding | 128 dimensions |
| Attention | Additive (Bahdanau) with source masking |
| Decoder | Unidirectional GRU (256 hidden) |
| Parameters | ~4 Million |
| Raw Size | ~15 MB |
| Quantized | ~4 MB (INT8) |

## Project Structure

```
TransliterationEngine/
├── src/                    # Core library
│   ├── config.py           # Constants & hyperparameters
│   ├── preprocessing.py    # Pre-model pipeline
│   ├── vocab.py            # Devanagari character vocabulary
│   ├── dataset.py          # PyTorch Dataset & DataLoader
│   ├── model.py            # Encoder, Attention, Decoder, Seq2Seq
│   ├── decoder.py          # Hybrid inference decoder
│   └── inference.py        # End-to-end inference wrapper
│
├── scripts/                # Training & evaluation
│   ├── download_data.py    # Phase 1: Data acquisition
│   ├── build_tokenizer.py  # Phase 2: BPE tokenizer training
│   ├── train_stage1.py     # Stage 1: Phonetics (20 epochs)
│   ├── train_stage2.py     # Stage 2: Word variations (10 epochs)
│   ├── train_stage3.py     # Stage 3: Sentence context (5 epochs)
│   ├── evaluate.py         # CER/WER/Top-k benchmarking
│   ├── export_onnx.py      # ONNX format export
│   ├── quantize_model.py   # INT8 quantization
│   └── benchmark_latency.py# Inference speed measurement
│
├── app/                    # Demo web application
│   ├── main.py             # FastAPI server
│   └── templates/index.html
│
├── tests/                  # Unit tests
├── notebooks/              # Original training notebooks
└── docs/                   # Documentation
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```python
from src.inference import TransliterationSystem

system = TransliterationSystem("transliteration_data/")
print(system.transliterate("namaste doston"))
# Output: नमस्ते दोस्तों
```

### Training Pipeline

```bash
# Phase 1: Download and prepare data
python scripts/download_data.py

# Phase 2: Build BPE tokenizer
python scripts/build_tokenizer.py

# Phase 4: 3-Stage curriculum training
python scripts/train_stage1.py   # Phonetics (20 epochs)
python scripts/train_stage2.py   # Word variations (10 epochs)
python scripts/train_stage3.py   # Sentence context (5 epochs)

# Phase 5: Evaluate
python scripts/evaluate.py

# Phase 6: Optimize for edge
python scripts/quantize_model.py
python scripts/export_onnx.py
python scripts/benchmark_latency.py
```

### Demo API

```bash
uvicorn app.main:app --reload
# Open http://localhost:8000 in your browser
```

## Training Curriculum

| Stage | Data | Epochs | Learning Rate | Goal |
|-------|------|--------|--------------|------|
| 1 | Aksharantar | 20 | 0.001 | Core phonetic mappings |
| 2 | Aksharantar + Dakshina | 10 | 0.0005 | Real-world spelling variations |
| 3 | Sentence Pairs | 5 | 0.0001 | Sentence-level context |

## Datasets

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| Aksharantar | AI4Bharat | ~1.5M pairs | Core phonetic learning |
| Dakshina Lexicons | Google | ~500K pairs | Spelling variation exposure |
| Dakshina Sentences | Google | ~2.5K pairs | Sentence-level context |

## Edge Optimization

- **Quantization**: FP32 → INT8 dynamic quantization (~9MB → ~4MB)
- **ONNX Export**: Separate encoder/decoder graphs for cross-platform deployment
- **Target Latency**: <50ms per sentence on mobile CPU

## Acknowledgements

- [AI4Bharat Aksharantar](https://ai4bharat.iitm.ac.in/aksharantar)
- [Google Dakshina Dataset](https://github.com/google-research-datasets/dakshina)
