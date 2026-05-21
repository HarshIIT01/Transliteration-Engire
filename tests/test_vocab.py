"""
Tests for the NativeVocab.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.vocab import NativeVocab
from src.config import SOS_IDX, EOS_IDX, PAD_IDX


class TestNativeVocab:
    """Test Devanagari character vocabulary."""

    def setup_method(self):
        self.vocab = NativeVocab()
        series = pd.Series(["नमस्ते", "भारत", "किसान"])
        self.vocab.build_vocab(series)

    def test_special_tokens_present(self):
        assert self.vocab.char2idx["[PAD]"] == 0
        assert self.vocab.char2idx["[UNK]"] == 1
        assert self.vocab.char2idx["[SOS]"] == 2
        assert self.vocab.char2idx["[EOS]"] == 3

    def test_vocab_size_correct(self):
        # 4 special + unique chars from नमस्ते, भारत, किसान
        assert self.vocab.vocab_size > 4

    def test_encode_has_sos_eos(self):
        encoded = self.vocab.encode("नमस्ते")
        assert encoded[0] == SOS_IDX
        assert encoded[-1] == EOS_IDX

    def test_decode_roundtrip(self):
        original = "भारत"
        encoded = self.vocab.encode(original)
        decoded = self.vocab.decode(encoded)
        assert decoded == original

    def test_additive_vocab(self):
        """Calling build_vocab again should extend, not overwrite."""
        old_size = self.vocab.vocab_size
        new_series = pd.Series(["ज़रूरत"])  # Has new chars like ज़, र, ू, त
        self.vocab.build_vocab(new_series)
        assert self.vocab.vocab_size >= old_size
