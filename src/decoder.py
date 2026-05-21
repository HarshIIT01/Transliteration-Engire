"""
Hybrid Inference Decoder
========================
Combines dictionary lookup with neural network inference
for production-quality transliteration.

Routing Logic:
    1. Dictionary Backoff (O(1) lookup) → for known common words
    2. Neural Greedy Decode → for out-of-vocabulary words
"""

import re
import torch

from src.config import SOS_IDX, EOS_IDX, MAX_DECODE_LEN, BEAM_WIDTH


class HybridDecoder:
    """
    Production-ready hybrid decoder that routes words through
    the fastest available path:
        - Common words → instant dictionary lookup
        - Rare/unknown words → neural network autoregressive decoding

    Args:
        model: Trained TransliterationEngine model (in eval mode).
        pipeline: PreModelPipeline instance with dictionary and tokenizer.
        trg_vocab: NativeVocab instance for decoding output indices.
        device: torch.device (cuda or cpu).
        max_word_len: Maximum number of characters to generate per word.
    """

    def __init__(self, model, pipeline, trg_vocab, device, max_word_len=MAX_DECODE_LEN):
        self.model = model
        self.pipeline = pipeline
        self.trg_vocab = trg_vocab
        self.device = device
        self.max_word_len = max_word_len

    def _decode_single_word_greedy(self, roman_word):
        """
        Neural Fallback: Greedy search (argmax) for OOV words.

        Generates one character at a time, always picking the
        highest-probability character until EOS is produced.

        Args:
            roman_word: Normalized Roman word string.

        Returns:
            Decoded Devanagari string.
        """
        encoded_src = self.pipeline.tokenizer.encode(roman_word).ids
        src_tensor = torch.tensor([encoded_src], dtype=torch.long).to(self.device)

        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)
            mask = self.model.create_mask(src_tensor)

            seq = [SOS_IDX]
            for _ in range(self.max_word_len):
                input_char = torch.tensor([seq[-1]]).to(self.device)
                output, hidden = self.model.decoder(
                    input_char, hidden, encoder_outputs, mask
                )

                top_pred = output.argmax(1).item()
                seq.append(top_pred)

                if top_pred == EOS_IDX:
                    break

            return self.trg_vocab.decode(seq)

    def _decode_single_word_beam(self, roman_word, beam_width=None):
        """
        Neural Fallback: Beam search for OOV words.

        Maintains `beam_width` candidate sequences in parallel,
        expanding each by the top-k next characters and keeping
        only the best `beam_width` overall. Uses length-normalized
        log-probabilities to avoid bias toward shorter outputs.

        Args:
            roman_word: Normalized Roman word string.
            beam_width: Number of beams (defaults to config BEAM_WIDTH).

        Returns:
            Decoded Devanagari string (best beam).
        """
        beam_width = beam_width or BEAM_WIDTH

        encoded_src = self.pipeline.tokenizer.encode(roman_word).ids
        src_tensor = torch.tensor([encoded_src], dtype=torch.long).to(self.device)

        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)
            mask = self.model.create_mask(src_tensor)

            # Each beam: (log_prob, sequence, hidden_state)
            beams = [(0.0, [SOS_IDX], hidden)]
            completed = []

            for _ in range(self.max_word_len):
                if not beams:
                    break

                all_candidates = []
                for score, seq, h in beams:
                    # If this beam already ended, move to completed
                    if seq[-1] == EOS_IDX:
                        completed.append((score, seq, h))
                        continue

                    input_char = torch.tensor([seq[-1]]).to(self.device)
                    output, new_h = self.model.decoder(
                        input_char, h, encoder_outputs, mask
                    )

                    log_probs = torch.log_softmax(output, dim=1).squeeze(0)
                    top_k_probs, top_k_ids = log_probs.topk(beam_width)

                    for i in range(beam_width):
                        new_score = score + top_k_probs[i].item()
                        new_seq = seq + [top_k_ids[i].item()]
                        all_candidates.append((new_score, new_seq, new_h))

                # Keep top beam_width candidates (normalized by length)
                all_candidates.sort(
                    key=lambda x: x[0] / max(len(x[1]) - 1, 1), reverse=True
                )
                beams = all_candidates[:beam_width]

                # Early exit if all beams completed
                if all(seq[-1] == EOS_IDX for _, seq, _ in beams):
                    completed.extend(beams)
                    break

            # Add any remaining beams to completed
            completed.extend(beams)

            if not completed:
                return ""

            # Select best by length-normalized score
            best = max(
                completed,
                key=lambda x: x[0] / max(len(x[1]) - 1, 1)
            )
            return self.trg_vocab.decode(best[1])

    def decode_word(self, roman_word, use_beam=True):
        """
        Decode a single word using the configured strategy.

        Args:
            roman_word: Normalized Roman word string.
            use_beam: If True, use beam search; otherwise greedy.

        Returns:
            Decoded Devanagari string.
        """
        if use_beam and BEAM_WIDTH > 1:
            return self._decode_single_word_beam(roman_word)
        return self._decode_single_word_greedy(roman_word)

    def transliterate(self, roman_text):
        """
        Transliterate a full Roman sentence into Devanagari.

        Routes each word through the optimal path:
            1. English word bypass (for loanwords kept in Roman)
            2. Dictionary lookup for known words (using normalized key)
            3. Neural decoding for unknown words

        Args:
            roman_text: Input Roman text (e.g., "namaste doston").

        Returns:
            Devanagari string (e.g., "नमस्ते दोस्तों").
        """
        # Split while preserving punctuation as separate tokens
        tokens = re.findall(r"[a-zA-Z]+|[^\sa-zA-Z]+|\s+", str(roman_text))
        native_parts = []

        for token in tokens:
            # Pass through whitespace
            if token.isspace():
                native_parts.append(" ")
                continue

            # Pass through punctuation/numbers as-is
            if not any(c.isalpha() for c in token):
                native_parts.append(token)
                continue

            w = token.lower()

            # ROUTE 1: English word bypass
            if self.pipeline.is_english(w):
                native_parts.append(token)
                continue

            # Normalize for lookup and neural inference
            norm_w = self.pipeline.normalize_roman(w)

            # ROUTE 2: O(1) Dictionary Lookup (using normalized key)
            if hasattr(self.pipeline, "fast_lookup") and norm_w in self.pipeline.fast_lookup:
                native_parts.append(self.pipeline.fast_lookup[norm_w])
                continue

            # ROUTE 3: Neural Network Fallback (beam search)
            native_pred = self.decode_word(norm_w)
            native_parts.append(native_pred)

        return "".join(native_parts)
