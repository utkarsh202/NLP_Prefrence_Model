"""Data preprocessing pipeline: cleans, randomizes A/B orders, applies IAA gates, and exports parquets."""
import argparse
from pathlib import Path
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
from src.utils.tokenization import format_input

ROOT = Path(__file__).resolve().parents[2]


def extract_prompt_and_response(text: str):
    marker = "\n\nAssistant:"
    last_idx = text.rfind(marker)
    if last_idx != -1:
        prompt = text[:last_idx + len(marker)].strip()
        response = text[last_idx + len(marker):].strip()
    else:
        parts = text.rsplit("Assistant:", 1)
        if len(parts) == 2:
            prompt = parts[0] + "Assistant:"
            response = parts[1].strip()
        else:
            prompt = ""
            response = text.strip()
    return prompt, response


def preprocess_hh(split_name="helpful", out_dir=None, seed=42):
    rng = random.Random(seed)
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=split_name)
    out_dir = out_dir or (ROOT / "data/processed/hh_rlhf_hard")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        records = []
        for row in ds[split]:
            pc, rc = extract_prompt_and_response(row["chosen"])
            pr, rr = extract_prompt_and_response(row["rejected"])
            prompt = pc if len(pc) > 0 else pr
            
            # 50/50 order randomization
            if rng.randint(0, 1) == 1:
                ra, rb, label = rc, rr, 1.0
            else:
                ra, rb, label = rr, rc, 0.0
                
            records.append({
                "prompt": prompt,
                "response_a": ra,
                "response_b": rb,
                "label": label,
                "len_a": len(ra.split()),
                "len_b": len(rb.split()),
                "subset": split_name
            })
            
        df = pd.DataFrame(records)
        if split == "train":
            # 95/5 train/val split
            val_n = min(6000, int(len(df) * 0.05))
            df.iloc[val_n:].to_parquet(out_dir / "train.parquet", index=False)
            df.iloc[:val_n].to_parquet(out_dir / "val.parquet", index=False)
        else:
            df.to_parquet(out_dir / "test.parquet", index=False)


def preprocess_shp(out_dir=None, seed=42):
    ds = load_dataset("stanfordnlp/SHP")
    out_dir = out_dir or (ROOT / "data/processed/shp_soft")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        records = []
        for row in ds[split]:
            sa, sb = float(row["score_A"]), float(row["score_B"])
            tot = sa + sb
            if tot <= 0:
                continue
            pa = sa / tot
            kappa = 2.0 * abs(pa - 0.5)

            # Quality gate on train only
            if split == "train" and (tot <= 20 or abs(sa - sb) <= 10 or kappa < 0.20):
                continue

            records.append({
                "prompt": row.get("history", row.get("post", "")),
                "response_a": row["human_ref_A"],
                "response_b": row["human_ref_B"],
                "p_a": pa,
                "label": 1.0 if pa >= 0.5 else 0.0,
                "high_disagreement": bool(0.4 < pa < 0.6),
                "len_a": len(row["human_ref_A"].split()),
                "len_b": len(row["human_ref_B"].split()),
                "domain": row.get("domain", "general")
            })

        df = pd.DataFrame(records)
        if split == "train":
            val_n = min(7000, int(len(df) * 0.08))
            df.iloc[val_n:].to_parquet(out_dir / "train.parquet", index=False)
            df.iloc[:val_n].to_parquet(out_dir / "val.parquet", index=False)
        else:
            df.to_parquet(out_dir / "test.parquet", index=False)


def preprocess_uf(out_dir=None, seed=42):
    rng = random.Random(seed)
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    out_dir = out_dir or (ROOT / "data/processed/ultrafeedback")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train_prefs", "test_prefs"]:
        records = []
        target_split = ds[split] if split in ds else ds.get("train", [])
        for row in target_split:
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")
            
            # Extract content if stored as chat message list
            if isinstance(chosen, list) and len(chosen) > 0:
                chosen = chosen[-1].get("content", str(chosen))
            if isinstance(rejected, list) and len(rejected) > 0:
                rejected = rejected[-1].get("content", str(rejected))

            # 50/50 order randomization
            if rng.randint(0, 1) == 1:
                ra, rb, label = chosen, rejected, 1.0
            else:
                ra, rb, label = rejected, chosen, 0.0

            records.append({
                "prompt": prompt,
                "response_a": str(ra),
                "response_b": str(rb),
                "label": label,
                "p_a": float(label),
                "len_a": len(str(ra).split()),
                "len_b": len(str(rb).split()),
                "subset": "ultrafeedback"
            })

        df = pd.DataFrame(records)
        out_name = "train" if "train" in split else "test"
        if out_name == "train":
            val_n = min(4000, int(len(df) * 0.1))
            df.iloc[val_n:].to_parquet(out_dir / "train.parquet", index=False)
            df.iloc[:val_n].to_parquet(out_dir / "val.parquet", index=False)
        else:
            df.to_parquet(out_dir / "test.parquet", index=False)


def main():
    parser = argparse.ArgumentParser(description="Preprocess preference datasets.")
    parser.add_argument("--datasets", nargs="+", default=["hh", "shp", "hhh", "uf"],
                        help="Datasets to process: hh, shp, hhh, uf")
    args = parser.parse_args()

    if "hh" in args.datasets:
        preprocess_hh("helpful")
    if "hhh" in args.datasets:
        preprocess_hh("harmless-base", ROOT / "data/processed/hh_harmless")
    if "shp" in args.datasets:
        preprocess_shp()
    if "uf" in args.datasets:
        preprocess_uf()

    print("[SUCCESS] Preprocessing completed.")


if __name__ == "__main__":
    main()
