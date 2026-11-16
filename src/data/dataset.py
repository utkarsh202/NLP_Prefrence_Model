from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.tokenization import build_input, MAX_PROMPT, MAX_RESP

ROOT = Path(__file__).resolve().parents[2]


def load_split(dataset, split, subset_size=None, seed=42):
    """
    Loads preprocessed dataset splits as pandas DataFrames.
    Maps dataset and split names to parquet files in data/processed/.
    """
    path_map = {
        ("hh", "train"): ROOT / "data/processed/hh_rlhf_hard/train.parquet",
        ("hh", "val"): ROOT / "data/processed/hh_rlhf_hard/val.parquet",
        ("hh", "test"): ROOT / "data/processed/hh_rlhf_hard/test.parquet",
        ("hh", "harmless_shift"): ROOT / "data/processed/hh_harmless/test.parquet",
        ("shp", "train"): ROOT / "data/processed/shp_soft/train.parquet",
        ("shp", "val"): ROOT / "data/processed/shp_soft/val.parquet",
        ("shp", "val_full"): ROOT / "data/processed/shp_soft/val.parquet",
        ("shp", "test"): ROOT / "data/processed/shp_soft/test.parquet",
        ("hhh", "train"): ROOT / "data/processed/hh_harmless/train.parquet",
        ("hhh", "val"): ROOT / "data/processed/hh_harmless/val.parquet",
        ("hhh", "test"): ROOT / "data/processed/hh_harmless/test.parquet",
        ("uf", "train"): ROOT / "data/processed/ultrafeedback/train.parquet",
        ("uf", "val"): ROOT / "data/processed/ultrafeedback/val.parquet",
        ("uf", "test"): ROOT / "data/processed/ultrafeedback/test.parquet",
    }
    key = (dataset, split)
    if key not in path_map:
        raise ValueError(f"Unknown split: {key}")
    
    file_path = path_map[key]
    if not file_path.exists():
        # Fallback to local data paths if root structure differs
        alt_paths = [
            ROOT / f"data/processed/{dataset}/{split}.parquet",
            ROOT / f"data/processed/{dataset}_{split}.parquet"
        ]
        for alt in alt_paths:
            if alt.exists():
                file_path = alt
                break

    if file_path.exists():
        df = pd.read_parquet(file_path)
    else:
        # Create empty placeholder DataFrame with required columns if file not on disk
        cols = ["prompt", "response_a", "response_b", "label", "len_a", "len_b"]
        if dataset == "shp":
            cols += ["p_a", "high_disagreement", "domain"]
        else:
            cols += ["subset"]
        df = pd.DataFrame(columns=cols)

    if subset_size and len(df) > subset_size:
        df = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)

    return df


class PairDataset(Dataset):
    """
    Dataset wrapping pairwise comparisons for Cross-Encoder and Bi-Encoder training.
    Supports swap-augmentation and token budgeting.
    """
    def __init__(self, df, tokenizer, label_type="hard", swap_prob=0.0,
                 max_prompt=MAX_PROMPT, max_resp=MAX_RESP):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.label_type = label_type
        self.swap_prob = swap_prob
        self.max_prompt = max_prompt
        self.max_resp = max_resp

        # Set target array for evaluation and loss
        if "p_a" in self.df.columns and label_type == "soft":
            self.target = self.df["p_a"].to_numpy(dtype=np.float32)
        elif "label" in self.df.columns:
            self.target = self.df["label"].to_numpy(dtype=np.float32)
        else:
            self.target = np.ones(len(self.df), dtype=np.float32)

        self.prompts = self.df["prompt"].tolist() if "prompt" in self.df else [""] * len(self.df)
        self.resps_a = self.df["response_a"].tolist() if "response_a" in self.df else [""] * len(self.df)
        self.resps_b = self.df["response_b"].tolist() if "response_b" in self.df else [""] * len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        a = self.resps_a[idx]
        b = self.resps_b[idx]
        target = float(self.target[idx])

        # Swap augmentation if requested during training
        if self.swap_prob > 0 and random.random() < self.swap_prob:
            a, b = b, a
            target = 1.0 - target

        enc = lambda t: self.tok.encode(t, add_special_tokens=False)
        p_ids = enc(prompt)
        a_ids = enc(a)
        b_ids = enc(b)

        # Cross input: [CLS] prompt [SEP] resp_a [SEP] resp_b [SEP]
        cross_inp = build_input(p_ids, a_ids, b_ids, self.tok, self.max_prompt, self.max_resp)

        # Siamese input:
        # resp A: [CLS] prompt [SEP] resp_a [SEP]
        # resp B: [CLS] prompt [SEP] resp_b [SEP]
        p_trunc = p_ids[-self.max_prompt:] if self.max_prompt > 0 else []
        a_trunc = a_ids[:self.max_resp]
        b_trunc = b_ids[:self.max_resp]
        cls_id = self.tok.cls_token_id
        sep_id = self.tok.sep_token_id

        ids_a = [cls_id] + p_trunc + [sep_id] + a_trunc + [sep_id]
        ids_b = [cls_id] + p_trunc + [sep_id] + b_trunc + [sep_id]

        return {
            "input_ids": cross_inp["input_ids"],
            "attention_mask": cross_inp["attention_mask"],
            "token_type_ids": cross_inp.get("token_type_ids", [0] * len(cross_inp["input_ids"])),
            "input_ids_a": ids_a,
            "attention_mask_a": [1] * len(ids_a),
            "input_ids_b": ids_b,
            "attention_mask_b": [1] * len(ids_b),
            "label": target
        }


def _pad_1d(sequences, pad_val=0):
    max_len = max(len(s) for s in sequences)
    padded = np.full((len(sequences), max_len), pad_val, dtype=np.int64)
    for i, s in enumerate(sequences):
        padded[i, :len(s)] = s
    return torch.tensor(padded, dtype=torch.long)


class CrossCollator:
    """Collates and dynamically pads inputs for CrossEncoder models."""
    def __init__(self, tokenizer, token_type=False):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.token_type = token_type

    def __call__(self, batch):
        out = {
            "input_ids": _pad_1d([b["input_ids"] for b in batch], self.pad_id),
            "attention_mask": _pad_1d([b["attention_mask"] for b in batch], 0),
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        }
        if self.token_type and "token_type_ids" in batch[0]:
            out["token_type_ids"] = _pad_1d([b["token_type_ids"] for b in batch], 0)
        return out


class SiameseCollator:
    """Collates and dynamically pads inputs for BiEncoder (Siamese) models."""
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        return {
            "input_ids_a": _pad_1d([b["input_ids_a"] for b in batch], self.pad_id),
            "attention_mask_a": _pad_1d([b["attention_mask_a"] for b in batch], 0),
            "input_ids_b": _pad_1d([b["input_ids_b"] for b in batch], self.pad_id),
            "attention_mask_b": _pad_1d([b["attention_mask_b"] for b in batch], 0),
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        }
