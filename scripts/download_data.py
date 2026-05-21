import os
import sys
import re
import glob
import tarfile
import zipfile
import urllib.request
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

# Must match src.config.DATA_DIR so Phase 2+ find the same CSVs and artifacts.
OUTPUT_DIR = DATA_DIR


def download_aksharantar():
    """Download and extract Aksharantar Hindi dataset."""
    print("1. Downloading & Extracting Aksharantar...")
    url = "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/hin.zip"
    zip_path = os.path.join(OUTPUT_DIR, "hin.zip")

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    extract_dir = os.path.join(OUTPUT_DIR, "aksharantar_hin_raw")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    json_files = glob.glob(os.path.join(extract_dir, "**/*.json"), recursive=True)

    dfs = []
    for file in json_files:
        try:
            try:
                df_part = pd.read_json(file)
            except ValueError:
                df_part = pd.read_json(file, lines=True)
            dfs.append(df_part)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]

    # Rename columns dynamically
    if "english word" in df.columns and "native word" in df.columns:
        df = df.rename(columns={"english word": "roman", "native word": "native"})
    elif "source" in df.columns and "target" in df.columns:
        df = df.rename(columns={"source": "roman", "target": "native"})

    df = df[["roman", "native"]]
    df["freq"] = 1
    print(f"-> Aksharantar pairs extracted: {len(df)}")
    return df


def download_dakshina():
    """Download and extract Dakshina Hindi lexicon dataset."""
    print("\n2. Downloading & Extracting Dakshina...")
    url = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
    tar_path = os.path.join(OUTPUT_DIR, "dakshina.tar")

    # Download with chunked approach (urlretrieve can silently corrupt large files)
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1_000_000:
        print("   Downloading Dakshina tar (~50 MB)...")
        if os.path.exists(tar_path):
            os.remove(tar_path)  # Remove any corrupted partial download
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response, open(tar_path, "wb") as out_f:
                while True:
                    chunk = response.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    out_f.write(chunk)
            print(f"   Downloaded: {os.path.getsize(tar_path) / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"   Download failed: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            raise

    # Verify the tar file is valid before extracting
    extract_dir = os.path.join(OUTPUT_DIR, "dakshina_raw")
    try:
        with tarfile.open(tar_path, "r:") as tar:
            if sys.version_info >= (3, 12):
                tar.extractall(extract_dir, filter="data")
            else:
                tar.extractall(extract_dir)
    except (tarfile.ReadError, EOFError) as e:
        print(f"   Tar file corrupted ({e}). Deleting and retrying...")
        os.remove(tar_path)
        return download_dakshina()  # Retry once

    base_path = os.path.join(extract_dir, "dakshina_dataset_v1.0", "hi", "lexicons")
    files = [
        "hi.translit.sampled.train.tsv",
        "hi.translit.sampled.dev.tsv",
        "hi.translit.sampled.test.tsv",
    ]

    dfs = []
    for file in files:
        try:
            df_part = pd.read_csv(
                os.path.join(base_path, file),
                sep="\t", header=None,
                names=["native", "roman", "freq"],
            )
            dfs.append(df_part)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    df = pd.concat(dfs, ignore_index=True)[["roman", "native", "freq"]]
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"-> Dakshina pairs extracted: {len(df)}")
    return df


def download_dakshina_sentences():
    """Extract Dakshina sentence pairs (requires Dakshina to be downloaded first)."""
    print("\n3. Extracting Dakshina Sentence Pairs...")
    base = os.path.join(OUTPUT_DIR, "dakshina_raw", "dakshina_dataset_v1.0", "hi", "romanized")

    dev_native = open(os.path.join(base, "hi.romanized.rejoined.dev.native.txt"), encoding="utf-8").read().splitlines()
    dev_roman = open(os.path.join(base, "hi.romanized.rejoined.dev.roman.txt"), encoding="utf-8").read().splitlines()
    test_native = open(os.path.join(base, "hi.romanized.rejoined.test.native.txt"), encoding="utf-8").read().splitlines()
    test_roman = open(os.path.join(base, "hi.romanized.rejoined.test.roman.txt"), encoding="utf-8").read().splitlines()

    dev_df = pd.DataFrame({"roman": dev_roman, "native": dev_native})
    test_df = pd.DataFrame({"roman": test_roman, "native": test_native})
    df = pd.concat([dev_df, test_df], ignore_index=True)
    print(f"-> Sentence pairs extracted: {len(df)}")
    return df


def is_devanagari(text):
    """Check if text contains only Devanagari characters and spaces."""
    for ch in text:
        if ch == " ":
            continue
        if not ("\u0900" <= ch <= "\u097F"):
            return False
    return True


def clean_and_aggregate(df, is_sentence=False):
    """Clean and aggregate transliteration pairs."""
    if df.empty:
        return df

    df = df.dropna()
    df["roman"] = df["roman"].astype(str).str.lower().str.strip()
    df["native"] = df["native"].astype(str).str.strip()
    df["native"] = df["native"].apply(lambda x: unicodedata.normalize("NFC", x))

    if is_sentence:
        df["roman"] = df["roman"].apply(lambda x: re.sub(r"[^a-z ]", " ", x))
        df["roman"] = df["roman"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
        df = df[df["native"].apply(is_devanagari)]
        df = df[df["roman"].str.len() >= 5]
        return df.drop_duplicates()
    else:
        df = df[df["roman"].str.match(r"^[a-z ]+$")]
        df = df[df["native"].str.match(r"^[\u0900-\u097F ]+$")]
        df = df[df["roman"].str.len() >= 2]
        df = df[df["native"].str.len() >= 2]

        if "freq" in df.columns:
            df = df.groupby(["roman", "native"], as_index=False)["freq"].sum()
        else:
            df = df.drop_duplicates(subset=["roman", "native"])

        return df


def split_and_save(df, prefix, output_dir):
    """Split dataset into train/val/test and save as CSV."""
    if len(df) == 0:
        return

    train, temp = train_test_split(df, test_size=0.1, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv(os.path.join(output_dir, f"{prefix}_train.csv"), index=False, encoding="utf-8")
    val.to_csv(os.path.join(output_dir, f"{prefix}_val.csv"), index=False, encoding="utf-8")
    test.to_csv(os.path.join(output_dir, f"{prefix}_test.csv"), index=False, encoding="utf-8")

    print(f"-> Saved {prefix.capitalize()}: Train ({len(train)}) | Val ({len(val)}) | Test ({len(test)})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_dir = os.path.join(OUTPUT_DIR, "master_corpus")
    sentences_dir = os.path.join(OUTPUT_DIR, "sentences")
    os.makedirs(master_dir, exist_ok=True)
    os.makedirs(sentences_dir, exist_ok=True)

    # Download datasets
    df_aksharantar = download_aksharantar()
    df_dakshina = download_dakshina()
    df_sentences = download_dakshina_sentences()

    # Clean independently
    print("\n4. Cleaning & Aggregating...")
    clean_aksharantar = clean_and_aggregate(df_aksharantar, is_sentence=False)
    clean_dakshina = clean_and_aggregate(df_dakshina, is_sentence=False)
    clean_sentences = clean_and_aggregate(df_sentences, is_sentence=True)

    print(f"-> Cleaned Aksharantar: {len(clean_aksharantar)}")
    print(f"-> Cleaned Dakshina: {len(clean_dakshina)}")
    print(f"-> Cleaned Sentences: {len(clean_sentences)}")

    print("\n5. Splitting and Saving...")
    split_and_save(clean_aksharantar, "aksharantar", OUTPUT_DIR)
    split_and_save(clean_dakshina, "dakshina", OUTPUT_DIR)
    split_and_save(clean_sentences, "sentences", sentences_dir)

    df_words = pd.concat([clean_aksharantar, clean_dakshina], ignore_index=True)
    split_and_save(df_words, "words", master_dir)

    print("\n Phase 1 Data Pipeline Complete!")


if __name__ == "__main__":
    main()
