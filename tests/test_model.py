"""
Tests for the neural model architecture.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import PAD_IDX
from src.model import Encoder, Decoder, TransliterationEngine, count_parameters


class TestModelArchitecture:
    """Test model compilation and forward pass."""

    def setup_method(self):
        self.device = torch.device("cpu")
        self.input_vocab = 5000
        self.output_vocab = 150
        self.batch_size = 4
        self.src_len = 10
        self.trg_len = 12

    def test_encoder_output_shape(self):
        enc = Encoder(self.input_vocab, pad_idx=PAD_IDX)
        src = torch.randint(1, self.input_vocab, (self.batch_size, self.src_len))
        outputs, hidden = enc(src)

        assert outputs.shape == (self.batch_size, self.src_len, 256 * 2)
        assert hidden.shape == (self.batch_size, 256)

    def test_full_forward_pass(self):
        enc = Encoder(self.input_vocab, pad_idx=PAD_IDX)
        dec = Decoder(self.output_vocab, pad_idx=PAD_IDX)
        model = TransliterationEngine(enc, dec, PAD_IDX, self.device)

        src = torch.randint(1, self.input_vocab, (self.batch_size, self.src_len))
        trg = torch.randint(1, self.output_vocab, (self.batch_size, self.trg_len))

        output = model(src, trg, teacher_forcing_ratio=0.5)
        assert output.shape == (self.batch_size, self.trg_len, self.output_vocab)

    def test_masking_works_with_padding(self):
        enc = Encoder(self.input_vocab, pad_idx=PAD_IDX)
        dec = Decoder(self.output_vocab, pad_idx=PAD_IDX)
        model = TransliterationEngine(enc, dec, PAD_IDX, self.device)

        src = torch.randint(1, self.input_vocab, (self.batch_size, self.src_len))
        src[:, -3:] = PAD_IDX  # Add padding at the end
        trg = torch.randint(1, self.output_vocab, (self.batch_size, self.trg_len))

        # Should not raise any errors
        output = model(src, trg)
        assert output.shape == (self.batch_size, self.trg_len, self.output_vocab)

    def test_parameter_count_reasonable(self):
        enc = Encoder(self.input_vocab, pad_idx=PAD_IDX)
        dec = Decoder(self.output_vocab, pad_idx=PAD_IDX)
        model = TransliterationEngine(enc, dec, PAD_IDX, self.device)

        params = count_parameters(model)
        # Should be in the ~2-5M range
        assert 1_000_000 < params < 10_000_000
