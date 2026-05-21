"""
Stage 2: Word-Level Variation Training
========================================
Fine-tunes the Stage 1 model on combined Aksharantar + Dakshina data
with frequency-weighted sampling to learn real-world spelling variations.

Improvements over original:
    - Label smoothing (0.1) for better generalization
    - ReduceLROnPlateau scheduler
    - Uses shared training utilities (no code duplication)

Usage:
    python scripts/train_stage2.py
"""

import os
import sys
import math
import json

import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PAD_IDX, DEVICE, BATCH_SIZE, GRAD_CLIP,
    STAGE2_CONFIG, DATA_DIR, MASTER_CORPUS_DIR, TOKENIZER_PATH,
)
from src.preprocessing import PreModelPipeline
from src.vocab import NativeVocab
from src.dataset import TransliterationDataset, collate_fn
from src.model import Encoder, Decoder, TransliterationEngine, count_parameters
from src.training_utils import (
    LabelSmoothingLoss, train_epoch, evaluate_epoch, sample_predictions,
)

from tokenizers import Tokenizer


def main():
    print(f"Stage 2 Training — Device: {DEVICE}\n")
    torch.backends.cudnn.benchmark = True

    # Rebuild pipeline
    corpus_paths = [
        os.path.join(DATA_DIR, "aksharantar_train.csv"),
        os.path.join(DATA_DIR, "dakshina_train.csv"),
    ]
    pipeline = PreModelPipeline(corpus_paths=corpus_paths, vocab_size=5000)

    if os.path.exists(TOKENIZER_PATH):
        pipeline.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # Rebuild vocabulary
    df_aksh = pd.read_csv(os.path.join(DATA_DIR, "aksharantar_train.csv"))
    trg_vocab = NativeVocab()
    trg_vocab.build_vocab(df_aksh["native"])

    INPUT_VOCAB_SIZE = pipeline.tokenizer.get_vocab_size()
    OUTPUT_VOCAB_SIZE = trg_vocab.vocab_size

    # Load Stage 2 data (combined words)
    words_train_path = os.path.join(MASTER_CORPUS_DIR, "words_train.csv")
    df_words_train = pd.read_csv(words_train_path)

    stage2_dataset = TransliterationDataset(df_words_train, pipeline, trg_vocab)

    # Frequency-weighted sampling
    weights = torch.DoubleTensor(df_words_train["freq"].values.copy())
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    stage2_loader = DataLoader(
        stage2_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # Validation data
    val_path = os.path.join(MASTER_CORPUS_DIR, "words_val.csv")
    df_val = pd.read_csv(val_path)
    val_dataset = TransliterationDataset(df_val, pipeline, trg_vocab)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # Load Stage 1 model
    print("\nLoading Stage 1 Weights...")
    enc = Encoder(INPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    dec = Decoder(OUTPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    model = TransliterationEngine(enc, dec, PAD_IDX, DEVICE).to(DEVICE)

    stage1_path = os.path.join(DATA_DIR, "stage1_best_model.pt")
    model.load_state_dict(torch.load(stage1_path, map_location=DEVICE, weights_only=True))
    print("-> Stage 1 weights loaded successfully!")

    # Lower learning rate to prevent catastrophic forgetting
    optimizer = optim.Adam(model.parameters(), lr=STAGE2_CONFIG["lr"], weight_decay=1e-5)
    criterion = LabelSmoothingLoss(smoothing=0.1, ignore_index=PAD_IDX)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True,
    )

    # Training loop with history logging
    config = STAGE2_CONFIG
    best_valid_loss = float("inf")
    save_path = os.path.join(DATA_DIR, config["save_name"])
    history_path = os.path.join(DATA_DIR, "stage2_history.json")

    sample_words = ["namaste", "bharat", "kolkata", "kisan", "sundar", "desh", "vidyalaya", "pariksha"]

    history = {
        "train_loss": [], "val_loss": [],
        "train_ppl": [], "val_ppl": [],
        "tf_ratio": [], "lr": [], "predictions": [],
    }

    print(f"\n--- Starting Stage 2 Training ({config['epochs']} Epochs) ---\n")

    try:
        for epoch in range(config["epochs"]):
            tf_ratio = max(config["tf_floor"], config["tf_start"] - epoch * config["tf_decay"])
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch+1:02} | TF Ratio: {tf_ratio:.2f} | LR: {current_lr:.6f}")

            train_loss = train_epoch(model, stage2_loader, optimizer, criterion, GRAD_CLIP, tf_ratio, DEVICE)
            valid_loss = evaluate_epoch(model, val_loader, criterion, DEVICE)

            # Step the LR scheduler
            scheduler.step(valid_loss)

            train_ppl = math.exp(min(train_loss, 10))
            valid_ppl = math.exp(min(valid_loss, 10))

            print(f"  Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}")
            print(f"  Val. Loss:  {valid_loss:.3f} | Val. PPL:  {valid_ppl:7.3f}")

            # Log metrics
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(valid_loss, 4))
            history["train_ppl"].append(round(train_ppl, 2))
            history["val_ppl"].append(round(valid_ppl, 2))
            history["tf_ratio"].append(round(tf_ratio, 4))
            history["lr"].append(current_lr)

            # Sample predictions every 3 epochs and epoch 1
            if (epoch + 1) % 3 == 0 or epoch == 0:
                preds = sample_predictions(model, pipeline, trg_vocab, sample_words, DEVICE)
                history["predictions"].append({"epoch": epoch + 1, "samples": preds})
                print(f"  Sample: {preds[0]['input']} → {preds[0]['prediction']}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), save_path)
                print(f"  [*] Best model saved to {save_path}")

            # Save history after every epoch
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Checkpoints remain saved.")

    print(f"\n✓ Stage 2 Training Complete!")
    print(f"  History saved to {history_path}")
    print(f"  Run: python scripts/visualize_training.py stage2")


if __name__ == "__main__":
    main()
