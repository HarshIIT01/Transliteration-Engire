import os
import sys
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, BPE_VOCAB_SIZE
from src.preprocessing import PreModelPipeline
from src.vocab import NativeVocab


def main():
    print("Phase 2: Building Tokenizer & Vocabulary\n")

    # Build the pipeline (trains BPE tokenizer + dictionary)
    corpus_paths = [
        os.path.join(DATA_DIR, "aksharantar_train.csv"),
        os.path.join(DATA_DIR, "dakshina_train.csv"),
    ]

    existing_paths = [p for p in corpus_paths if os.path.exists(p)]
    if not existing_paths:
        print("ERROR: No training data found. Run scripts/download_data.py first.")
        return

    pipeline = PreModelPipeline(corpus_paths=existing_paths, vocab_size=BPE_VOCAB_SIZE)

    # Save the tokenizer
    tokenizer_path = os.path.join(DATA_DIR, "tokenizer.json")
    pipeline.tokenizer.save(tokenizer_path)
    print(f"\n-> Tokenizer saved to {tokenizer_path}")

    # Build and save target vocabulary
    print("\nBuilding Target Vocabulary...")
    df_aksh = pd.read_csv(os.path.join(DATA_DIR, "aksharantar_train.csv"))

    trg_vocab = NativeVocab()
    trg_vocab.build_vocab(df_aksh["native"])

    # Also include sentence data if available
    sent_path = os.path.join(DATA_DIR, "sentences", "sentences_train.csv")
    if os.path.exists(sent_path):
        df_sent = pd.read_csv(sent_path)
        trg_vocab.build_vocab(df_sent["native"])

    vocab_path = os.path.join(DATA_DIR, "trg_vocab.pt")
    torch.save(trg_vocab, vocab_path)
    print(f"-> Target vocabulary saved to {vocab_path} ({trg_vocab.vocab_size} chars)")

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
