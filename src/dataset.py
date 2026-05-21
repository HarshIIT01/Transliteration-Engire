import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.config import PAD_IDX


class TransliterationDataset(Dataset):
    def __init__(self, df, pipeline, target_vocab):
        self.df = df[["roman", "native"]].dropna().reset_index(drop=True)
        self.pipeline = pipeline
        self.trg_vocab = target_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_roman = str(self.df.iloc[idx]["roman"])
        native_word = str(self.df.iloc[idx]["native"])

        # Normalize exactly as inference does
        norm_roman = self.pipeline.normalize_roman(raw_roman)

        # Encode Source (Roman) using BPE Tokenizer
        encoded_src = self.pipeline.tokenizer.encode(norm_roman).ids
        src_tensor = torch.tensor(encoded_src, dtype=torch.long)

        # Encode Target (Devanagari) using Character Vocab (with SOS/EOS)
        encoded_trg = self.trg_vocab.encode(native_word)
        trg_tensor = torch.tensor(encoded_trg, dtype=torch.long)

        return src_tensor, trg_tensor


def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        src_batch.append(src_item)
        trg_batch.append(trg_item)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)

    return src_padded, trg_padded
