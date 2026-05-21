"""
Phase 6: ONNX Export
=====================
Exports the trained Encoder and Decoder as separate ONNX graphs
for cross-platform deployment (mobile, C++, etc).

Usage:
    python scripts/export_onnx.py
"""

import os
import sys

import torch
import pandas as pd
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PAD_IDX, DEVICE, DATA_DIR, TOKENIZER_PATH, ENC_HID_DIM
from src.vocab import NativeVocab
from src.model import Encoder, Decoder, TransliterationEngine


def _load_trg_vocab():
    """Load saved target vocab or rebuild from training CSV."""
    trg_vocab_path = os.path.join(DATA_DIR, "trg_vocab.pt")
    if os.path.exists(trg_vocab_path):
        trg_vocab = torch.load(trg_vocab_path, map_location="cpu", weights_only=False)
        print(f"-> Target vocab loaded from {trg_vocab_path} ({trg_vocab.vocab_size} chars)")
        return trg_vocab

    trg_vocab = NativeVocab()
    aksh_path = os.path.join(DATA_DIR, "aksharantar_train.csv")
    if not os.path.exists(aksh_path):
        raise FileNotFoundError(
            f"No trg_vocab.pt or aksharantar_train.csv in {DATA_DIR}. "
            "Run build_tokenizer.py first."
        )
    df_aksh = pd.read_csv(aksh_path)
    trg_vocab.build_vocab(df_aksh["native"])
    print(f"-> Target vocab built from CSV ({trg_vocab.vocab_size} chars)")
    return trg_vocab


def _load_model_weights(model):
    """Load best available checkpoint; handle vocab size mismatches."""
    candidates = [
        os.path.join(DATA_DIR, "stage3_final_model.pt"),
        os.path.join(DATA_DIR, "stage2_best_model.pt"),
        os.path.join(DATA_DIR, "stage1_best_model.pt"),
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if not model_path:
        raise FileNotFoundError(
            f"No model weights found in {DATA_DIR}. Train at least stage 1 first."
        )

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    ckpt_vocab_size = state_dict["decoder.embedding.weight"].shape[0]
    model_vocab_size = model.decoder.output_vocab_size

    if ckpt_vocab_size != model_vocab_size:
        print(
            f"-> Vocab surgery: checkpoint {ckpt_vocab_size} chars, "
            f"model {model_vocab_size} chars"
        )
        new_state_dict = model.state_dict()
        for name, param in state_dict.items():
            if name not in new_state_dict:
                continue
            if new_state_dict[name].shape == param.shape:
                new_state_dict[name] = param
            elif name in (
                "decoder.embedding.weight",
                "decoder.fc_out.weight",
                "decoder.fc_out.bias",
            ):
                new_state_dict[name][: param.shape[0]] = param
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    print(f"-> Model weights loaded from {os.path.basename(model_path)}")
    return model_path


def main():
    print("Phase 6: ONNX Export\n")

    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: pip install onnx onnxscript\n"
            "PyTorch needs the 'onnx' package to write .onnx files."
        ) from e

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}. Run build_tokenizer.py first.")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    input_vocab_size = tokenizer.get_vocab_size()

    trg_vocab = _load_trg_vocab()
    output_vocab_size = trg_vocab.vocab_size

    enc = Encoder(input_vocab_size, pad_idx=PAD_IDX)
    dec = Decoder(output_vocab_size, pad_idx=PAD_IDX)
    model = TransliterationEngine(enc, dec, PAD_IDX, DEVICE).to(DEVICE)
    _load_model_weights(model)
    model.eval()

    # ==========================================
    # EXPORT ENCODER
    # ==========================================
    print("Exporting Encoder to ONNX...")
    dummy_src = torch.randint(1, input_vocab_size, (1, 10), device=DEVICE)

    encoder_path = os.path.join(DATA_DIR, "encoder.onnx")
    torch.onnx.export(
        model.encoder,
        dummy_src,
        encoder_path,
        input_names=["source"],
        output_names=["encoder_outputs", "hidden"],
        dynamic_axes={
            "source": {0: "batch", 1: "src_len"},
            "encoder_outputs": {0: "batch", 1: "src_len"},
            "hidden": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    encoder_size = os.path.getsize(encoder_path) / (1024 * 1024)
    print(f"-> Encoder saved: {encoder_path} ({encoder_size:.2f} MB)")

    # ==========================================
    # EXPORT DECODER (single step)
    # ==========================================
    print("\nExporting Decoder to ONNX...")
    dummy_input_char = torch.tensor([2], device=DEVICE)
    dummy_hidden = torch.randn(1, ENC_HID_DIM, device=DEVICE)
    dummy_enc_out = torch.randn(1, 10, ENC_HID_DIM * 2, device=DEVICE)
    dummy_mask = torch.ones(1, 10, dtype=torch.bool, device=DEVICE)

    decoder_path = os.path.join(DATA_DIR, "decoder.onnx")
    torch.onnx.export(
        model.decoder,
        (dummy_input_char, dummy_hidden, dummy_enc_out, dummy_mask),
        decoder_path,
        input_names=["input_char", "hidden", "encoder_outputs", "mask"],
        output_names=["prediction", "new_hidden"],
        dynamic_axes={
            "encoder_outputs": {0: "batch", 1: "src_len"},
            "mask": {0: "batch", 1: "src_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    decoder_size = os.path.getsize(decoder_path) / (1024 * 1024)
    print(f"-> Decoder saved: {decoder_path} ({decoder_size:.2f} MB)")

    total_size = encoder_size + decoder_size
    print(f"\nTotal ONNX size: {total_size:.2f} MB")
    print("ONNX export complete.")


if __name__ == "__main__":
    main()
