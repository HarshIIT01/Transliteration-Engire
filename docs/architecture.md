# System Architecture

## Overview

The Transliteration Engine converts Roman (Latin) script into Devanagari (Hindi) script while preserving pronunciation. It uses a hybrid approach combining instant dictionary lookup with a neural seq2seq model.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT PIPELINE                         │
│                                                          │
│   Roman Input → Language Detection → Roman Normalization │
│                      │                      │            │
│              English words              aa→a, ph→f       │
│              bypass (copy)              ee→i, oo→u       │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │ Dictionary      │  O(1) lookup
              │ Backoff         │  Top 50K words
              │ (Hash Map)      │
              └────────┬────────┘
                       │ Miss
              ┌────────▼────────┐
              │ BPE Tokenizer   │  Subword encoding
              │ (5K vocab)      │  Preserves conjuncts
              └────────┬────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  NEURAL MODEL (~4M params)               │
│                                                          │
│   ┌─────────────┐     ┌───────────┐   ┌──────────────┐   │
│   │   Encoder    │    │ Attention │   │   Decoder    │   │
│   │ BiGRU (256)  │──▶│ Bahdanau  │──▶│  GRU (256)   │   │
│   │ + Embedding  │    │ + Masking │   │ + Softmax    │   │
│   │   (128-dim)  │    │           │   │              │   │
│   └─────────────-┘    └───────────┘   └──────────────┘   │
└──────────────────────┬──────────────────────────────────-┘
                       │
              ┌────────▼────────┐
              │ Devanagari      │
              │ Output          │
              └─────────────────┘
```

## Training Curriculum (3-Stage)

```
Stage 1: Phonetics (20 epochs, LR=0.001)
├── Data: Aksharantar only
├── Goal: Learn basic mappings (k→क, m→म)
└── TF Ratio: 0.7 → 0.3

    ▼ Load Stage 1 weights

Stage 2: Variations (10 epochs, LR=0.0005)
├── Data: Aksharantar + Dakshina
├── Goal: Handle real-world spelling chaos
├── Sampling: Frequency-weighted
└── TF Ratio: 0.4 → 0.1

    ▼ Load Stage 2 weights + Vocab Surgery

Stage 3: Sentences (5 epochs, LR=0.0001)
├── Data: Dakshina Sentence Pairs
├── Goal: Sentence-level context understanding
├── Vocab Expansion: Add space + punctuation chars
└── TF Ratio: 0.2 → 0.0
```

## Edge Deployment Pipeline

```
Trained Model (FP32, ~15MB)
    │
    ├── Dynamic Quantization (INT8) → ~4MB
    │
    ├── ONNX Export → encoder.onnx + decoder.onnx
    │
    └── FastAPI Wrapper → REST API for mobile integration
```

## Key Design Decisions

1. **BiGRU over Transformer**: GRU is lightweight (~4M params vs 100M+), trainable on a single GPU, and fast enough for real-time keyboard use.

2. **Hybrid Inference**: Dictionary lookup handles the most common words instantly (O(1)), while the neural network handles rare/unseen words. This dramatically reduces average latency.

3. **Curriculum Learning**: Progressive training prevents catastrophic forgetting — Stage 1 phonetics are locked in before introducing noisy real-world data.

4. **Source Masking**: PAD tokens are masked before softmax in the attention layer to prevent the model from attending to padding positions.

5. **Vocabulary Surgery**: Stage 3 expands the target vocabulary (to include spaces for sentences) while preserving all Stage 1/2 learned weights via careful tensor slicing.
