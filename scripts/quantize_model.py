"""
Phase 6: INT8 Dynamic Quantization
====================================
Quantizes the trained model from FP32 to INT8 for edge deployment.
Targets GRU and Linear layers for maximum compression.

Usage:
    python scripts/quantize_model.py
"""

import os
import sys

import torch
import torch.nn as nn
import pandas as pd
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PAD_IDX, DATA_DIR, TOKENIZER_PATH
from src.vocab import NativeVocab
from src.model import Encoder, Decoder, TransliterationEngine


def get_model_size_mb(model):
    """Calculate model size in MB by saving to a temporary file."""
    tmp_path = os.path.join(DATA_DIR, "_tmp_model.pt")
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


def main():
    print("Phase 6: INT8 Dynamic Quantization\n")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    INPUT_VOCAB_SIZE = tokenizer.get_vocab_size()

    # Load vocabulary
    df_aksh = pd.read_csv(os.path.join(DATA_DIR, "aksharantar_train.csv"))
    trg_vocab = NativeVocab()
    trg_vocab.build_vocab(df_aksh["native"])
    OUTPUT_VOCAB_SIZE = trg_vocab.vocab_size

    # Load model on CPU (quantization requires CPU)
    device = torch.device("cpu")
    enc = Encoder(INPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    dec = Decoder(OUTPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    model = TransliterationEngine(enc, dec, PAD_IDX, device).to(device)

    model_path = os.path.join(DATA_DIR, "stage2_best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Measure original size
    original_size = get_model_size_mb(model)
    print(f"Original Model Size: {original_size:.2f} MB")

    # Apply dynamic quantization targeting GRU and Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.GRU, nn.Linear},
        dtype=torch.qint8,
    )

    # Measure quantized size
    quantized_path = os.path.join(DATA_DIR, "stage2_quantized.pt")
    torch.save(quantized_model.state_dict(), quantized_path)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)

    compression_ratio = original_size / quantized_size

    print(f"Quantized Model Size: {quantized_size:.2f} MB")
    print(f"Compression Ratio: {compression_ratio:.1f}x")
    print(f"\nSaved to: {quantized_path}")

    # Quick validation - ensure the quantized model still works
    print("\nValidating quantized model...")
    dummy_src = torch.randint(1, INPUT_VOCAB_SIZE, (1, 10))
    dummy_trg = torch.randint(1, OUTPUT_VOCAB_SIZE, (1, 8))

    with torch.no_grad():
        output = quantized_model(dummy_src, dummy_trg, teacher_forcing_ratio=0)
    print(f"Forward pass successful. Output shape: {output.shape}")

    print("\n✓ Quantization Complete!")


if __name__ == "__main__":
    main()
