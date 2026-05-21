"""
Phase 6: Latency Benchmarking
===============================
Measures inference latency for PyTorch FP32, Quantized INT8,
and (optionally) ONNX Runtime models.

Usage:
    python scripts/benchmark_latency.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
import pandas as pd
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PAD_IDX, SOS_IDX, EOS_IDX, DATA_DIR, TOKENIZER_PATH, MAX_DECODE_LEN
from src.preprocessing import PreModelPipeline
from src.vocab import NativeVocab
from src.model import Encoder, Decoder, TransliterationEngine
from src.decoder import HybridDecoder


def benchmark_model(decoder, test_inputs, label, num_runs=3):
    """Benchmark a decoder on test inputs, averaging over multiple runs."""
    # Warmup
    for text in test_inputs[:2]:
        decoder.transliterate(text)

    total_time = 0
    for _ in range(num_runs):
        start = time.perf_counter()
        for text in test_inputs:
            decoder.transliterate(text)
        end = time.perf_counter()
        total_time += (end - start)

    avg_total = (total_time / num_runs) * 1000
    avg_per = avg_total / len(test_inputs)

    return {"label": label, "total_ms": avg_total, "per_input_ms": avg_per}


def main():
    print("Phase 6: Latency Benchmark\n")
    device = torch.device("cpu")  # Benchmarking on CPU (simulating mobile)

    # Load pipeline
    corpus_paths = [os.path.join(DATA_DIR, "master_corpus", "words_train.csv")]
    existing = [p for p in corpus_paths if os.path.exists(p)]
    if not existing:
        corpus_paths = [
            os.path.join(DATA_DIR, "aksharantar_train.csv"),
            os.path.join(DATA_DIR, "dakshina_train.csv"),
        ]
        existing = [p for p in corpus_paths if os.path.exists(p)]

    pipeline = PreModelPipeline(corpus_paths=existing, vocab_size=5000)
    pipeline.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    INPUT_VOCAB_SIZE = pipeline.tokenizer.get_vocab_size()

    # Load vocabulary
    df_aksh = pd.read_csv(os.path.join(DATA_DIR, "aksharantar_train.csv"))
    trg_vocab = NativeVocab()
    trg_vocab.build_vocab(df_aksh["native"])
    OUTPUT_VOCAB_SIZE = trg_vocab.vocab_size

    # Test inputs
    test_inputs = [
        "namaste",
        "mera naam harshit hai",
        "bharat ek sundar desh hai",
        "vah vishuddh chaitanya swarup ki or agrasar ho raha hota hai",
        "kya kar rahe ho",
    ]

    results = []

    # ==========================================
    # 1. PyTorch FP32
    # ==========================================
    print("Benchmarking PyTorch FP32...")
    enc = Encoder(INPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    dec = Decoder(OUTPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    model_fp32 = TransliterationEngine(enc, dec, PAD_IDX, device).to(device)
    model_fp32.load_state_dict(
        torch.load(os.path.join(DATA_DIR, "stage2_best_model.pt"),
                    map_location=device, weights_only=True)
    )
    model_fp32.eval()

    decoder_fp32 = HybridDecoder(model_fp32, pipeline, trg_vocab, device)
    result = benchmark_model(decoder_fp32, test_inputs, "PyTorch FP32")
    results.append(result)

    # ==========================================
    # 2. Quantized INT8
    # ==========================================
    print("Benchmarking Quantized INT8...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.GRU, nn.Linear}, dtype=torch.qint8
    )

    decoder_int8 = HybridDecoder(model_int8, pipeline, trg_vocab, device)
    result = benchmark_model(decoder_int8, test_inputs, "Quantized INT8")
    results.append(result)

    # ==========================================
    # 3. ONNX Runtime (if available)
    # ==========================================
    encoder_onnx_path = os.path.join(DATA_DIR, "encoder.onnx")
    if os.path.exists(encoder_onnx_path):
        try:
            import onnxruntime as ort
            print("Benchmarking ONNX Runtime...")
            # Note: Full ONNX autoregressive inference would require
            # a custom loop with ONNX sessions. This is a placeholder
            # for demonstrating the framework.
            print("  (ONNX autoregressive benchmark requires custom session loop)")
            print("  Run export_onnx.py first, then integrate with onnxruntime.InferenceSession")
        except ImportError:
            print("  onnxruntime not installed, skipping ONNX benchmark")

    # ==========================================
    # RESULTS TABLE
    # ==========================================
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Total (ms)':<15} {'Per Input (ms)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<20} {r['total_ms']:<15.2f} {r['per_input_ms']:<15.2f}")
    print("=" * 60)

    # Check target
    target_ms = 50
    for r in results:
        status = "✓ PASS" if r["per_input_ms"] < target_ms else "✗ FAIL"
        print(f"{r['label']}: {r['per_input_ms']:.2f}ms/sentence — {status} (target: <{target_ms}ms)")

    print("\n✓ Benchmark Complete!")


if __name__ == "__main__":
    main()
