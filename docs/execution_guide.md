# Execution Guide

Step-by-step guide for running the Transliteration Engine project. Each step lists **what you run**, **what it calls internally**, and **what it produces**.

---

## Pipeline Overview

```
download_data.py → build_tokenizer.py → train_stage1.py → train_stage2.py → train_stage3.py → evaluate.py
                                                                                                    ↓
                                                                              export_onnx.py / quantize_model.py / benchmark_latency.py
                                                                                                    ↓
                                                                                          uvicorn app.main:app
```

---

## Step 1: Download & Prepare Data

```bash
python scripts/download_data.py
```

| What it does | What it calls internally |
|---|---|
| Downloads Aksharantar (Hindi ZIP from HuggingFace) | `urllib.request.urlretrieve` |
| Downloads Dakshina (TAR from Google Storage) | `tarfile.open` + `extractall` |
| Extracts Dakshina sentence pairs | Reads `.txt` files from `dakshina_raw/` |
| Cleans all datasets (regex, Unicode NFC, dedup) | `clean_and_aggregate()` |
| Splits each into Train/Val/Test (90/5/5) | `sklearn.train_test_split` |

**Produces:**
```
transliteration_data/
├── aksharantar_train.csv, aksharantar_val.csv, aksharantar_test.csv
├── dakshina_train.csv, dakshina_val.csv, dakshina_test.csv
├── sentences/
│   ├── sentences_train.csv, sentences_val.csv, sentences_test.csv
└── master_corpus/
    └── words_train.csv, words_val.csv, words_test.csv   (aksharantar + dakshina merged)
```

---

## Step 2: Build Tokenizer & Vocabulary

```bash
python scripts/build_tokenizer.py
```

| What it does | What it calls internally |
|---|---|
| Loads aksharantar + dakshina training CSVs | `pandas.read_csv` |
| Builds the Pre-Model Pipeline (normalizer + dictionary + BPE) | `src.preprocessing.PreModelPipeline` |
| Trains a BPE tokenizer (5000 vocab) on normalized Roman text | `tokenizers.BpeTrainer` |
| Builds Devanagari character vocabulary from all training data | `src.vocab.NativeVocab.build_vocab()` |

**Produces:**
```
transliteration_data/
├── tokenizer.json      ← BPE tokenizer (Roman input encoding)
└── trg_vocab.pt        ← Devanagari char vocab (target decoding)
```

---

## Step 3: Stage 1 Training — Phonetics

```bash
python scripts/train_stage1.py
```

| What it does | What it calls internally |
|---|---|
| Loads `aksharantar_train.csv` + `aksharantar_val.csv` | `src.dataset.TransliterationDataset` |
| Creates DataLoaders with dynamic padding | `src.dataset.collate_fn` |
| Initializes fresh Encoder + Decoder + TransliterationEngine | `src.model.Encoder`, `src.model.Decoder` |
| Trains 20 epochs with teacher forcing decay (0.7 → 0.3) | `train_epoch()` loop |
| Validates after each epoch, saves best model | `torch.save(model.state_dict())` |

**Produces:**
```
transliteration_data/stage1_best_model.pt    ← ~9MB model weights
```

---

## Step 4: Stage 2 Training — Word Variations

```bash
python scripts/train_stage2.py
```

| What it does | What it calls internally |
|---|---|
| Loads `master_corpus/words_train.csv` (aksharantar + dakshina combined) | `pandas.read_csv` |
| Creates a `WeightedRandomSampler` based on `freq` column | `torch.utils.data.WeightedRandomSampler` |
| **Loads Stage 1 weights** into a fresh model | `model.load_state_dict(torch.load("stage1_best_model.pt"))` |
| Trains 10 epochs with lower LR (0.0005) to prevent forgetting | `optim.Adam(lr=0.0005)` |

**Depends on:** `stage1_best_model.pt` from Step 3

**Produces:**
```
transliteration_data/stage2_best_model.pt    ← ~9MB model weights
```

---

## Step 5: Stage 3 Training — Sentence Context

```bash
python scripts/train_stage3.py
```

| What it does | What it calls internally |
|---|---|
| Loads `sentences/sentences_train.csv` | `pandas.read_csv` |
| Expands target vocabulary (adds space + punctuation chars) | `NativeVocab.build_vocab()` called twice (additive) |
| **Loads Stage 2 weights with vocabulary surgery** — copies old weights, initializes new vocab entries randomly | Manual `state_dict` tensor slicing |
| Trains 5 epochs with lowest LR (0.0001) | `optim.Adam(lr=0.0001)` |
| Re-saves expanded `trg_vocab.pt` | `torch.save(trg_vocab)` |

**Depends on:** `stage2_best_model.pt` from Step 4

**Produces:**
```
transliteration_data/stage3_final_model.pt   ← ~9MB model weights
transliteration_data/trg_vocab.pt            ← updated (expanded vocab)
```

---

## Step 6: Evaluate

```bash
python scripts/evaluate.py
```

| What it does | What it calls internally |
|---|---|
| Loads the trained model + pipeline + dictionary | `src.inference` components |
| Creates a `HybridDecoder` (dictionary lookup + neural fallback) | `src.decoder.HybridDecoder` |
| Runs inference on 5000 test samples | `decoder.transliterate()` per sample |
| Computes CER, WER, Exact Match Accuracy | `character_error_rate()`, `word_error_rate()` |
| Prints formatted results table | stdout |

**Depends on:** `stage2_best_model.pt`, `tokenizer.json`, training CSVs

---

## Step 7: Edge Optimization (Phase 6)

These three scripts can be run **in any order** after training is complete:

### 7a: Quantize Model
```bash
python scripts/quantize_model.py
```
| Does | Calls |
|---|---|
| Loads FP32 model on CPU | `torch.load()` |
| Applies INT8 dynamic quantization to GRU + Linear layers | `torch.quantization.quantize_dynamic()` |
| Validates forward pass still works | dummy tensor test |

**Produces:** `transliteration_data/stage2_quantized.pt` (~3-4MB)

### 7b: Export to ONNX
```bash
python scripts/export_onnx.py
```
| Does | Calls |
|---|---|
| Exports Encoder as ONNX graph with dynamic axes | `torch.onnx.export(model.encoder)` |
| Exports Decoder as separate ONNX graph (autoregressive) | `torch.onnx.export(model.decoder)` |

**Produces:** `transliteration_data/encoder.onnx` + `decoder.onnx`

### 7c: Benchmark Latency
```bash
python scripts/benchmark_latency.py
```
| Does | Calls |
|---|---|
| Benchmarks FP32 vs INT8 on 5 test sentences | `time.perf_counter()` |
| Prints comparison table and PASS/FAIL vs 50ms target | stdout |

---

## Step 8: Run Demo App

```bash
uvicorn app.main:app --reload
```

| What it does | What it calls internally |
|---|---|
| Starts FastAPI server on `http://localhost:8000` | `uvicorn` |
| On startup, loads the full `TransliterationSystem` | `src.inference.TransliterationSystem(DATA_DIR)` |
| `GET /` serves the web UI | `app/templates/index.html` |
| `POST /transliterate` accepts `{"text": "..."}` and returns Devanagari + latency | `system.transliterate()` |

**Open `http://localhost:8000`** in your browser → type Roman text → see live Devanagari output.

---

## Step 9: Run Tests

```bash
python -m pytest tests/ -v
```

| Test File | What it tests |
|---|---|
| `test_preprocessing.py` | Roman normalization rules, English word detection |
| `test_vocab.py` | Vocab build, encode/decode roundtrip, additive expansion |
| `test_model.py` | Forward pass shapes, masking with PAD, parameter count |
| `test_inference.py` | End-to-end transliteration (skipped if model weights absent) |

---

## Quick Reference: Dependency Chain

```
download_data.py
    └─→ build_tokenizer.py
            └─→ train_stage1.py
                    └─→ train_stage2.py
                            └─→ train_stage3.py
                            └─→ evaluate.py
                            └─→ quantize_model.py
                            └─→ export_onnx.py
                            └─→ benchmark_latency.py
                            └─→ uvicorn app.main:app
```

Each arrow means "depends on outputs from the previous step".
