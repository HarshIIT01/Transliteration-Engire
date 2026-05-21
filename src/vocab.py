"""
Character-level vocabulary builder for the target script.
"""

from src.config import PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX

class NativeVocab:
    """
    Builds and manages a character-level vocabulary for Devanagari script.

    Special tokens:
        [PAD] = 0, [UNK] = 1, [SOS] = 2, [EOS] = 3

    Usage:
        vocab = NativeVocab()
        vocab.build_vocab(df['native'])       # Build from pandas Series
        encoded = vocab.encode("नमस्ते")      # [2, 4, 5, 6, 7, 8, 3]
        decoded = vocab.decode(encoded)        # "नमस्ते"
    """

    def __init__(self):
        self.char2idx = {
            "[PAD]": PAD_IDX,
            "[UNK]": UNK_IDX,
            "[SOS]": SOS_IDX,
            "[EOS]": EOS_IDX,
        }
        self.idx2char = {
            PAD_IDX: "[PAD]",
            UNK_IDX: "[UNK]",
            SOS_IDX: "[SOS]",
            EOS_IDX: "[EOS]",
        }
        self.vocab_size = 4

    def build_vocab(self, series):
        """
        Build vocabulary from a pandas Series of native-script strings.

        This method is additive — calling it multiple times extends the
        vocabulary without overwriting existing mappings. This is critical
        for the curriculum training approach where Stage 3 adds space and
        punctuation characters on top of Stage 1/2 Devanagari characters.

        Args:
            series: pandas Series containing native script strings.
        """
        print("Building Devanagari Character Vocabulary...")
        unique_chars = set()
        for word in series.dropna():
            unique_chars.update(list(str(word)))

        for char in sorted(list(unique_chars)):
            if char not in self.char2idx:
                self.char2idx[char] = self.vocab_size
                self.idx2char[self.vocab_size] = char
                self.vocab_size += 1

        print(f"-> Target Vocabulary built with {self.vocab_size} unique characters.")

    def encode(self, word):
        
        chars = list(str(word))
        encoded = [self.char2idx.get(c, UNK_IDX) for c in chars]
        return [SOS_IDX] + encoded + [EOS_IDX]

    def decode(self, indices):
        
        chars = []
        for idx in indices:
            if isinstance(idx, int):
                token_idx = idx
            else:
                token_idx = idx.item()

            if token_idx in (PAD_IDX, SOS_IDX):
                continue
            if token_idx == EOS_IDX:
                break
            chars.append(self.idx2char.get(token_idx, "[UNK]"))
        return "".join(chars)
