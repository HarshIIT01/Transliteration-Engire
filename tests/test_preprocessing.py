"""
Tests for the PreModelPipeline.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import PreModelPipeline


class TestNormalization:
    """Test the Roman normalization layer."""

    def setup_method(self):
        self.pipeline = PreModelPipeline(corpus_paths=None)

    def test_double_vowel_preserved(self):
        """Double vowels encode long vowels in Hindi and must be preserved."""
        assert self.pipeline.normalize_roman("aa") == "aa"   # आ
        assert self.pipeline.normalize_roman("ee") == "ee"   # ई
        assert self.pipeline.normalize_roman("oo") == "oo"   # ऊ

    def test_triple_vowel_collapsed(self):
        """3+ repeated vowels are typos and should collapse to double."""
        assert self.pipeline.normalize_roman("aaaa") == "aa"
        assert self.pipeline.normalize_roman("eee") == "ee"
        assert self.pipeline.normalize_roman("ooo") == "oo"

    def test_ph_preserved(self):
        """'ph' is the Romanization of फ and must NOT become 'f'."""
        assert self.pipeline.normalize_roman("phool") == "phool"
        assert self.pipeline.normalize_roman("phone") == "phone"

    def test_lowercase_strip(self):
        assert self.pipeline.normalize_roman("  NAMASTE  ") == "namaste"

    def test_special_chars_removed(self):
        assert self.pipeline.normalize_roman("hello123!") == "hello"

    def test_hindi_words_preserved(self):
        """Common Hindi words should pass through normalization intact."""
        assert self.pipeline.normalize_roman("raat") == "raat"    # रात
        assert self.pipeline.normalize_roman("keel") == "keel"    # कील
        assert self.pipeline.normalize_roman("naam") == "naam"    # नाम
        assert self.pipeline.normalize_roman("baat") == "baat"    # बात

    def test_consonant_triple_collapsed(self):
        """3+ repeated consonants are typos."""
        assert self.pipeline.normalize_roman("hellooo") == "helloo"
        assert self.pipeline.normalize_roman("nmmm") == "nmm"


class TestLanguageDetection:
    """Test the English word bypass."""

    def setup_method(self):
        self.pipeline = PreModelPipeline(corpus_paths=None)

    def test_tech_loanwords_detected(self):
        """Tech loanwords in the allowlist should be bypassed."""
        assert self.pipeline.is_english("laptop") is True
        assert self.pipeline.is_english("internet") is True
        assert self.pipeline.is_english("camera") is True

    def test_hindi_words_not_detected(self):
        """Common Hindi Romanizations must NOT be falsely bypassed."""
        assert self.pipeline.is_english("naam") is False    # नाम - previously broken!
        assert self.pipeline.is_english("baat") is False    # बात - previously broken!
        assert self.pipeline.is_english("raat") is False    # रात - previously broken!
        assert self.pipeline.is_english("desh") is False    # देश - previously broken!
        assert self.pipeline.is_english("namaste") is False
        assert self.pipeline.is_english("kisan") is False
        assert self.pipeline.is_english("pani") is False    # पानी

    def test_unknown_words_treated_as_hindi(self):
        """Unknown words that aren't in allowlist should be treated as Hindi."""
        assert self.pipeline.is_english("xyzabc") is False
        assert self.pipeline.is_english("harshit") is False
