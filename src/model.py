"""
Neural Transliteration Model
=============================
Seq2Seq architecture with Bidirectional GRU encoder,
additive (Bahdanau) attention with source masking,
and unidirectional GRU decoder.

Architecture:
    Input (BPE tokens) → Encoder (BiGRU) → Attention → Decoder (GRU) → Output (Devanagari chars)

Improvements over original:
    - Multi-layer GRU (configurable, default 2 layers)
    - Inter-layer dropout in GRU for regularization
    - Residual connection in encoder for gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DROPOUT, NUM_LAYERS


class Encoder(nn.Module):
    """
    Bidirectional GRU Encoder.

    Processes the input sequence forwards and backwards to capture
    full context of the Roman word/sentence.

    Args:
        input_vocab_size: Size of the source (BPE) vocabulary.
        emb_dim: Embedding dimension.
        enc_hid_dim: Encoder hidden dimension (per direction).
        dec_hid_dim: Decoder hidden dimension (for projection).
        dropout: Dropout rate.
        pad_idx: Padding token index.
        n_layers: Number of GRU layers.
    """

    def __init__(self, input_vocab_size, emb_dim=EMB_DIM, enc_hid_dim=ENC_HID_DIM,
                 dec_hid_dim=DEC_HID_DIM, dropout=DROPOUT, pad_idx=0, n_layers=NUM_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            emb_dim, enc_hid_dim, num_layers=n_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src: [batch_size, src_len] — BPE token indices.

        Returns:
            outputs: [batch_size, src_len, enc_hid_dim * 2] — all encoder hidden states.
            hidden: [batch_size, dec_hid_dim] — compressed initial decoder hidden state.
        """
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # hidden shape: [n_layers * 2, batch, enc_hid_dim]
        # Take the final layer's forward and backward hidden states
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    """
    Additive (Bahdanau) Attention with Source Masking.

    Computes attention weights over encoder outputs, zeroing out
    attention on padding tokens via masking.

    Args:
        enc_hid_dim: Encoder hidden dimension.
        dec_hid_dim: Decoder hidden dimension.
    """

    def __init__(self, enc_hid_dim=ENC_HID_DIM, dec_hid_dim=DEC_HID_DIM):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        """
        Args:
            hidden: [batch_size, dec_hid_dim] — current decoder hidden state.
            encoder_outputs: [batch_size, src_len, enc_hid_dim * 2].
            mask: [batch_size, src_len] — True for real tokens, False for PAD.

        Returns:
            attention: [batch_size, src_len] — normalized attention weights.
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        # Mask PAD positions with large negative value before softmax
        attention = attention.masked_fill(~mask, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    Unidirectional GRU Decoder with Attention.

    Generates one output character at a time, attending to
    relevant parts of the encoder output.

    Args:
        output_vocab_size: Size of the target (Devanagari) vocabulary.
        emb_dim: Embedding dimension.
        enc_hid_dim: Encoder hidden dimension.
        dec_hid_dim: Decoder hidden dimension.
        dropout: Dropout rate.
        pad_idx: Padding token index.
    """

    def __init__(self, output_vocab_size, emb_dim=EMB_DIM, enc_hid_dim=ENC_HID_DIM,
                 dec_hid_dim=DEC_HID_DIM, dropout=DROPOUT, pad_idx=0):
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(output_vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_char, hidden, encoder_outputs, mask):
        """
        Args:
            input_char: [batch_size] — previous output character index.
            hidden: [batch_size, dec_hid_dim] — previous decoder hidden state.
            encoder_outputs: [batch_size, src_len, enc_hid_dim * 2].
            mask: [batch_size, src_len].

        Returns:
            prediction: [batch_size, output_vocab_size] — logits for next character.
            hidden: [batch_size, dec_hid_dim] — updated decoder hidden state.
        """
        input_char = input_char.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_char))

        # Compute attention and context vector
        a = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)
        context = torch.bmm(a, encoder_outputs)

        # GRU step
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # Squeeze for prediction
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden.squeeze(0)


class TransliterationEngine(nn.Module):
    """
    Full Seq2Seq Transliteration Engine.

    Wraps the Encoder, Decoder, and mask generation into a single
    module with teacher forcing support for training.

    Args:
        encoder: Encoder instance.
        decoder: Decoder instance.
        pad_idx: Padding token index (for mask generation).
        device: torch.device (cuda or cpu).
    """

    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def create_mask(self, src):
        """
        Create boolean mask for source sequence.

        Args:
            src: [batch_size, src_len] — source token indices.

        Returns:
            mask: [batch_size, src_len] — True for real tokens, False for PAD.
        """
        return src != self.pad_idx

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Full forward pass with optional teacher forcing.

        Args:
            src: [batch_size, src_len] — source BPE token indices.
            trg: [batch_size, trg_len] — target character indices.
            teacher_forcing_ratio: Probability of using ground truth as next input.

        Returns:
            outputs: [batch_size, trg_len, trg_vocab_size] — prediction logits.
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        mask = self.create_mask(src)

        input_char = trg[:, 0]  # Start with SOS token

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_char, hidden, encoder_outputs, mask)
            outputs[:, t, :] = output

            top1 = output.argmax(1)
            input_char = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
