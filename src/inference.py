"""
End-to-End Inference Wrapper
=============================
High-level API that loads all artifacts (model, tokenizer, vocabulary,
dictionary) and provides a simple one-liner transliteration interface.

Usage:
    from src.inference import TransliterationSystem

    system = TransliterationSystem("transliteration_data/")
    result = system.transliterate("namaste doston")
    print(result)  # नमस्ते दोस्तों
"""

import os
import json
import torch
from tokenizers import Tokenizer

from src.config import PAD_IDX, DEVICE
from src.preprocessing import PreModelPipeline
from src.vocab import NativeVocab
from src.model import Encoder, Decoder, TransliterationEngine
from src.decoder import HybridDecoder


class TransliterationSystem:
    """
    Complete transliteration system that loads all trained artifacts
    and provides a ready-to-use API.

    Args:
        data_dir: Path to the directory containing model weights,
                  tokenizer.json, trg_vocab.pt, and training CSVs.
        model_file: Name of the model weights file to load.
                    Defaults to "stage2_best_model.pt" (word-level model).
        device: torch.device (defaults to auto-detected DEVICE from config).
    """

    def __init__(self, data_dir, model_file="stage2_best_model.pt", device=None):
        self.device = device or DEVICE
        self.data_dir = data_dir

        print("Loading Transliteration System...")

        # 1. Load the pre-model pipeline with dictionary backoff
        #    Priority: JSON export (fast) > master CSV > individual CSVs
        dict_json_path = os.path.join(data_dir, "dictionary_backoff.json")

        if os.path.exists(dict_json_path):
            # FAST PATH: Load pre-exported dictionary (~2s boot)
            self.pipeline = PreModelPipeline(corpus_paths=None, vocab_size=5000)
            with open(dict_json_path, "r", encoding="utf-8") as f:
                self.pipeline.fast_lookup = json.load(f)
            print(f"-> Dictionary Backoff loaded from JSON: {len(self.pipeline.fast_lookup)} entries")
        else:
            # SLOW PATH: Rebuild from CSVs (~15s boot)
            master_word_path = os.path.join(data_dir, "master_corpus", "words_train.csv")
            if os.path.exists(master_word_path):
                self.pipeline = PreModelPipeline(corpus_paths=[master_word_path], vocab_size=5000)
            else:
                paths = [
                    os.path.join(data_dir, "aksharantar_train.csv"),
                    os.path.join(data_dir, "dakshina_train.csv"),
                ]
                existing_paths = [p for p in paths if os.path.exists(p)]
                self.pipeline = PreModelPipeline(corpus_paths=existing_paths, vocab_size=5000)
            print("  (Tip: Run 'python scripts/export_dictionary.py' to speed up future boots)")

        # 2. Load the saved tokenizer (locked to training tokenizer)
        tokenizer_path = os.path.join(data_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            self.pipeline.tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f"-> Tokenizer loaded from {tokenizer_path}")

        input_vocab_size = self.pipeline.tokenizer.get_vocab_size()

        # 3. Load the target vocabulary
        trg_vocab_path = os.path.join(data_dir, "trg_vocab.pt")
        if os.path.exists(trg_vocab_path):
            self.trg_vocab = torch.load(trg_vocab_path, map_location=self.device, weights_only=False)
            print(f"-> Target vocab loaded ({self.trg_vocab.vocab_size} chars)")
        else:
            # Rebuild from aksharantar training data
            import pandas as pd
            self.trg_vocab = NativeVocab()
            aksh_path = os.path.join(data_dir, "aksharantar_train.csv")
            if os.path.exists(aksh_path):
                df = pd.read_csv(aksh_path)
                self.trg_vocab.build_vocab(df["native"])

        output_vocab_size = self.trg_vocab.vocab_size

        # 4. Build and load the model
        #    Auto-detect best available checkpoint first, then build model to match
        model_candidates = [
            os.path.join(data_dir, "stage3_final_model.pt"),
            os.path.join(data_dir, model_file),
            os.path.join(data_dir, "stage2_best_model.pt"),
            os.path.join(data_dir, "stage1_best_model.pt"),
        ]
        model_path = next((p for p in model_candidates if os.path.exists(p)), None)

        if model_path:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

            # Auto-detect encoder layer count from checkpoint keys
            # PyTorch BiGRU keys: encoder.rnn.weight_ih_l0, encoder.rnn.weight_ih_l0_reverse,
            #                     encoder.rnn.weight_ih_l1, encoder.rnn.weight_ih_l1_reverse, ...
            # Count unique layer indices from non-reverse keys
            import re as _re
            layer_indices = set()
            for key in state_dict.keys():
                m = _re.match(r"encoder\.rnn\.weight_ih_l(\d+)$", key)
                if m:
                    layer_indices.add(int(m.group(1)))
            ckpt_n_layers = max(len(layer_indices), 1)
            print(f"-> Checkpoint uses {ckpt_n_layers}-layer encoder")

            # Build model to match checkpoint architecture
            enc = Encoder(input_vocab_size, pad_idx=PAD_IDX, n_layers=ckpt_n_layers)
            ckpt_vocab_size = state_dict["decoder.embedding.weight"].shape[0]

            # Build decoder with max of checkpoint and target vocab size
            dec_vocab = max(ckpt_vocab_size, output_vocab_size)
            dec = Decoder(dec_vocab, pad_idx=PAD_IDX)
            self.model = TransliterationEngine(enc, dec, PAD_IDX, self.device).to(self.device)

            # Weight surgery if vocab sizes differ
            if ckpt_vocab_size != dec_vocab:
                print(f"-> Vocab surgery: checkpoint has {ckpt_vocab_size}, model needs {dec_vocab}")
                new_state_dict = self.model.state_dict()
                for name, param in state_dict.items():
                    if name in new_state_dict:
                        if new_state_dict[name].shape == param.shape:
                            new_state_dict[name] = param
                        elif name in ["decoder.embedding.weight", "decoder.fc_out.weight", "decoder.fc_out.bias"]:
                            new_state_dict[name][:param.shape[0]] = param
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)

            print(f"-> Model weights loaded from {os.path.basename(model_path)}")
        else:
            # No checkpoint — build with default config layers
            enc = Encoder(input_vocab_size, pad_idx=PAD_IDX)
            dec = Decoder(output_vocab_size, pad_idx=PAD_IDX)
            self.model = TransliterationEngine(enc, dec, PAD_IDX, self.device).to(self.device)
            print(f"[WARNING] No model weights found. Model is uninitialized.")

        self.model.eval()

        # 5. Create the hybrid decoder
        self.decoder = HybridDecoder(
            self.model, self.pipeline, self.trg_vocab, self.device
        )

        print("Transliteration System ready!\n")

    def transliterate(self, text):
        """
        Transliterate Roman text to Devanagari.

        Args:
            text: Input Roman text (e.g., "mera naam sujal hai").

        Returns:
            Devanagari string (e.g., "मेरा नाम सुजल है").
        """
        return self.decoder.transliterate(text)
