#!/bin/bash
# hpc_launch.sh — SLURM script for IIT Bhilai GPU cluster (Plan Section 8.2)
#SBATCH --job-name=pref-deberta-large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_deberta_large.out

set -euo pipefail

# Navigate to project root
cd "$(dirname "$0")/../.."

mkdir -p logs

module load python/3.10 cuda/11.8 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

python -m src.training.train \
  --config configs/deberta_large.yaml \
  --device cuda \
  --gradient_checkpointing true \
  --fp16 true \
  --seed 42
