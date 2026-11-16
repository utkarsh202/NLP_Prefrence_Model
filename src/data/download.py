"""Data download CLI: downloads Anthropic HH-RLHF, Stanford SHP, and UltraFeedback from Hugging Face."""
import argparse
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]


def download_datasets(datasets):
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    if "hh" in datasets:
        print("Downloading Anthropic/hh-rlhf (helpful and harmless)...")
        load_dataset("Anthropic/hh-rlhf")
        
    if "shp" in datasets:
        print("Downloading stanfordnlp/SHP...")
        load_dataset("stanfordnlp/SHP")
        
    if "uf" in datasets:
        print("Downloading HuggingFaceH4/ultrafeedback_binarized...")
        load_dataset("HuggingFaceH4/ultrafeedback_binarized")

    print("[SUCCESS] All requested datasets fetched.")


def main():
    parser = argparse.ArgumentParser(description="Download preference datasets from Hugging Face.")
    parser.add_argument("--datasets", nargs="+", default=["hh", "shp", "uf"],
                        help="Datasets to download: hh, shp, uf")
    args = parser.parse_args()
    download_datasets(args.datasets)


if __name__ == "__main__":
    main()
