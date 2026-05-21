"""
Training Utilities
===================
Shared training loop components and loss functions used across
all 3 curriculum training stages.

Eliminates code duplication and provides:
    - LabelSmoothingLoss: Smoothed cross-entropy for better generalization
    - train_epoch / evaluate_epoch: Standard training loops
    - sample_predictions: Quick inference for progress monitoring
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Instead of one-hot targets (100% on correct class), distributes
    a small fraction `smoothing` uniformly across all classes. This
    prevents the model from becoming overconfident and improves
    generalization to unseen words.

    Args:
        smoothing: Fraction of probability to spread (default 0.1).
        ignore_index: Token index to ignore in loss (PAD).
    """

    def __init__(self, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] logits (N = batch * seq_len, C = vocab_size)
            target: [N] class indices
        """
        vocab_size = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)

        # Create smoothed target distribution
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (vocab_size - 2))
            smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
            # Zero out PAD positions
            mask = target == self.ignore_index
            smooth_target[mask] = 0

        loss = (-smooth_target * log_probs).sum(dim=-1)
        # Only average over non-PAD positions
        non_pad = (~mask).float()
        if non_pad.sum() > 0:
            loss = (loss * non_pad).sum() / non_pad.sum()
        else:
            loss = loss.sum()

        return loss


def train_epoch(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio, device):
    """
    Run a single training epoch.

    Args:
        model: TransliterationEngine model.
        iterator: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        clip: Gradient clipping max norm.
        teacher_forcing_ratio: Probability of using ground truth.
        device: torch.device.

    Returns:
        Average epoch loss.
    """
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(iterator, leave=False, desc="Training")

    for src, trg in progress_bar:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(iterator)


def evaluate_epoch(model, iterator, criterion, device):
    """
    Run a single evaluation epoch.

    Args:
        model: TransliterationEngine model.
        iterator: DataLoader for validation data.
        criterion: Loss function.
        device: torch.device.

    Returns:
        Average epoch loss.
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def sample_predictions(model, pipeline, trg_vocab, test_words, device):
    """
    Generate sample predictions for a list of test words.

    Args:
        model: TransliterationEngine model (set to eval internally).
        pipeline: PreModelPipeline instance.
        trg_vocab: NativeVocab instance.
        test_words: List of Roman words/sentences to test.
        device: torch.device.

    Returns:
        List of dicts with 'input' and 'prediction' keys.
    """
    from src.decoder import HybridDecoder
    decoder = HybridDecoder(model, pipeline, trg_vocab, device)
    model.eval()
    results = []
    for word in test_words:
        pred = decoder.transliterate(word)
        results.append({"input": word, "prediction": pred})
    return results
