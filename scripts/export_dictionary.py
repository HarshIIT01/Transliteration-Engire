"""
Export Dictionary Backoff to JSON
=================================
Builds the 50K roman→native dictionary from training CSVs
and saves it as a compact JSON file for fast server boot.

Usage:
    python scripts/export_dictionary.py

Output:
    transliteration_data/dictionary_backoff.json  (~2 MB)
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR
from src.preprocessing import PreModelPipeline


def main():
    print("=" * 50)
    print("  Exporting Dictionary Backoff to JSON")
    print("=" * 50)

    # Build pipeline from training data
    master_word_path = os.path.join(DATA_DIR, "master_corpus", "words_train.csv")
    if os.path.exists(master_word_path):
        paths = [master_word_path]
        print(f"\n  Source: {master_word_path}")
    else:
        paths = [
            os.path.join(DATA_DIR, "aksharantar_train.csv"),
            os.path.join(DATA_DIR, "dakshina_train.csv"),
        ]
        paths = [p for p in paths if os.path.exists(p)]
        print(f"\n  Sources: {len(paths)} CSV files")

    pipeline = PreModelPipeline(corpus_paths=paths, vocab_size=5000)

    # Export dictionary
    out_path = os.path.join(DATA_DIR, "dictionary_backoff.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pipeline.fast_lookup, f, ensure_ascii=False, indent=0)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n  [OK] Exported {len(pipeline.fast_lookup)} entries to:")
    print(f"    {out_path}")
    print(f"    Size: {size_mb:.2f} MB")
    print(f"\n  This replaces the need for 40MB+ training CSVs at deploy time.")


if __name__ == "__main__":
    main()
