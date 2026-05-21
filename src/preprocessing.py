import re
import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class PreModelPipeline:
    def __init__(self, corpus_paths=None, vocab_size=5000):
        self.fast_lookup = {}
        self.tokenizer = None
        if corpus_paths is None:
            return

        self._build_pipeline(corpus_paths, vocab_size)

    def normalize_roman(self, text):
        """
        Normalize Roman text while preserving phonetically significant patterns.
        
        Hindi-critical preservations:
            - 'aa' → 'aa' (long आ: raat, baat, naam)
            - 'ee' → 'ee' (long ई: neel, keel, cheeni)  
            - 'oo' → 'oo' (long ऊ: phool, mool, dhool)
            - 'ph' → 'ph' (फ phoneme: phool, phir, phone)
        
        Only collapse 3+ repeated vowels (genuine typos):
            - 'aaaa' → 'aa', 'eeee' → 'ee', 'oooo' → 'oo'
        """
        word = str(text).lower().strip()

        # Only collapse 3+ repeated vowels to 2 (preserve genuine doubles)
        word = re.sub(r"a{3,}", "aa", word)
        word = re.sub(r"e{3,}", "ee", word)
        word = re.sub(r"i{3,}", "ii", word)
        word = re.sub(r"o{3,}", "oo", word)
        word = re.sub(r"u{3,}", "uu", word)

        # Collapse 3+ repeated consonants to 2 (typo correction)
        word = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1{2,}", r"\1\1", word)

        # Do NOT convert ph→f: 'ph' is the standard Romanization of फ
        # Do NOT collapse aa→a, ee→i, oo→u: these encode long vowels

        word = re.sub(r"[^a-z]", "", word)
        return word

    def _build_pipeline(self, corpus_paths, vocab_size):
        print("1. Loading training datasets to build Phase 2 modules...")
        dfs = []
        for path in corpus_paths:
            try:
                dfs.append(pd.read_csv(path))
            except FileNotFoundError:
                print(f"Warning: {path} not found. Skipping.")

        if not dfs:
            print("CRITICAL ERROR: No data found to build dictionary and tokenizer.")
            return

        df = pd.concat(dfs, ignore_index=True)

        print("2. Normalizing corpus to build Dictionary Backoff...")
        df["roman_norm"] = df["roman"].apply(self.normalize_roman)

        # STEP 3: Dictionary Backoff
        if "freq" in df.columns:
            df_freq = df.groupby(["roman_norm", "native"], as_index=False)["freq"].sum()
        else:
            df_freq = df.groupby(["roman_norm", "native"]).size().reset_index(name="freq")

        df_freq = df_freq.sort_values("freq", ascending=False)

        top_50k = df_freq.drop_duplicates(subset=["roman_norm"], keep="first").head(50000)
        self.fast_lookup = top_50k.set_index("roman_norm")["native"].to_dict()
        print(f"-> Dictionary Backoff built: {len(self.fast_lookup)} Top-K words loaded.")

        # STEP 4: Subword Tokenization (BPE)
        print("3. Training Byte-Pair Encoding (BPE) Tokenizer...")
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
        )

        normalized_texts = df["roman_norm"].dropna().tolist()
        self.tokenizer.train_from_iterator(normalized_texts, trainer=trainer)
        print(f"-> Subword Tokenizer trained. Vocab size: {self.tokenizer.get_vocab_size()}")

    # Curated allowlist of loanwords that should always stay in Roman script.
    # We use an allowlist (not the full NLTK corpus) to avoid false positives
    # on Hindi words like 'naam', 'baat', 'raat', 'desh', 'kal' that also
    # happen to exist in English dictionaries.
    ENGLISH_LOANWORDS = {
        "wifi", "laptop", "phone", "charge", "internet", "app", "browser",
        "computer", "email", "google", "download", "upload", "software",
        "hardware", "website", "online", "video", "audio", "camera",
        "selfie", "WhatsApp", "facebook", "youtube", "twitter", "instagram",
        "password", "username", "account", "profile", "android", "iphone",
        "okay", "ok", "hello", "bye", "thanks", "sorry", "please",
        "school", "college", "office", "doctor", "hospital", "police",
    }

    def is_english(self, word):
        """
        Returns True ONLY if the word should be kept in Roman script (not transliterated).

        Priority rules:
            1. If the word is in our Hindi training dictionary → always Hindi (return False).
            2. If the word is in our curated English loanword allowlist → English (return True).
            3. Everything else → treat as Hindi (return False).

        We intentionally do NOT use the full NLTK English corpus here because it
        contains thousands of false positives for Romanized Hindi words:
            'naam' (नाम), 'baat' (बात), 'raat' (रात), 'desh' (देश),
            'kal' (कल), 'aaj' (आज), 'pani' (पानी), etc.
        """
        w = word.lower()
        norm_w = self.normalize_roman(w)

        # Rule 1: If this word is in our Hindi dictionary, it's definitely Hindi
        if norm_w in self.fast_lookup:
            return False

        # Rule 2: Curated tech/loanword allowlist
        if w in self.ENGLISH_LOANWORDS or word in self.ENGLISH_LOANWORDS:
            return True

        return False

    def process_word(self, word):
        original_word = str(word).strip()
        clean_word = original_word.lower()

        if self.is_english(clean_word):
            return {
                "word": original_word,
                "route": "English Bypass",
                "output": original_word,
                "tokens": None,
            }

        norm_word = self.normalize_roman(clean_word)

        if norm_word in self.fast_lookup:
            return {
                "word": original_word,
                "route": "Dictionary",
                "output": self.fast_lookup[norm_word],
                "tokens": None,
            }

        if self.tokenizer:
            encoded = self.tokenizer.encode(norm_word)
            return {
                "word": original_word,
                "route": "Neural Model",
                "output": "-> [PENDING INFERENCE]",
                "tokens": encoded.tokens,
            }

        return {"word": original_word, "route": "Error", "output": None, "tokens": None}

    def process_sentence(self, sentence):
        words_in_sentence = sentence.split()
        return [self.process_word(w) for w in words_in_sentence]
