# Reliable Pairwise Preference Modeling for LLM Responses
## Improved Project Plan — Version 2.0

> **Institution:** IIT Bhilai | **Domain:** NLP / LLM Alignment  
> **Duration:** 6 Weeks | **Author:** Utkarsh  
> **Hardware:** Apple M4 (16 GB) + IIT Bhilai HPC GPU Cluster  
> **Status:** Revised based on feasibility review

---

## Executive Summary

We propose a 6-week project to build and rigorously evaluate a **pairwise preference model** that predicts which of two LLM-generated responses a human would prefer. The core contribution is not raw prediction accuracy but **reliability, calibration, and bias awareness** — three properties critical for RLHF reward model deployment.

**Key improvements over the original proposal:**
- Crowdsourcing plan replaced with **Stanford Human Preferences (SHP)** dataset, providing real multi-annotator vote ratios for soft-label training at zero cost.
- Explicit **token budget strategy** defined upfront for all transformer models.
- Compute responsibilities clearly split between **Mac M4 (local dev)** and **IIT Bhilai HPC** (large-scale training).
- **Weights & Biases (W&B)** experiment tracking integrated from Day 1.
- Concrete **IAA thresholds** and data quality gates defined.
- **LLM judge baseline pinned** to GPT-4 and LLaMA-3-8B for reproducible comparison.

---

## 1. Research Questions & Hypotheses

### Primary Research Questions

**RQ1 — Accuracy vs. Model Scale:**
> *Can a fine-tuned, small transformer encoder (DeBERTa-v3-base, ~125M params) reliably predict human pairwise preferences, compared to prompting larger LLM judges like GPT-4-turbo and LLaMA-3-8B?*

**Pinned Baselines for RQ1:**
| Judge | Method | Size |
| :--- | :--- | :--- |
| GPT-4-turbo | Zero-shot prompt: "Which response is better, A or B?" | ~1.7T (proprietary) |
| LLaMA-3-8B (GGUF Q4) | Few-shot prompt with 3 examples | 8B (local inference on M4) |
| Our Cross-Encoder | Fine-tuned DeBERTa-v3-base | 125M |

---

**RQ2 — Non-Content Bias Sensitivity:**
> *How sensitive is the trained preference model to order-of-presentation (positional bias) and response length (verbosity bias)? What is the magnitude of these biases, and do calibration techniques reduce them?*

**Measurable Hypothesis:** A model trained without explicit debiasing will show >10% accuracy flip rate on the Swap Test (i.e., it changes its prediction when A and B are reversed). After debiasing / calibration, this should drop below 5%.

---

**RQ3 — Soft-Label Calibration:**
> *Does training with real annotator vote-fraction distributions (soft labels) from SHP — rather than hard majority-vote labels — improve calibration (ECE) and uncertainty estimates on held-out data?*

**Measurable Hypothesis:** The model trained with soft labels will achieve an ECE ≤ 0.08, versus ≥ 0.15 for the hard-label trained model, on the SHP test split with high disagreement pairs.

---

## 2. Datasets

### 2.1 Primary Dataset: Anthropic HH-RLHF

| Property | Value |
| :--- | :--- |
| Source | `Anthropic/hh-rlhf` on Hugging Face |
| Size | ~169,000 pairwise comparisons |
| Format | JSON: `{prompt, chosen, rejected}` |
| Label Type | **Hard binary** (chosen = preferred) |
| Content Warning | Red-teaming split contains toxic/offensive content |
| Usage in Project | Training & evaluating Accuracy, AUC, Swap Test, Length Bias |

**Preprocessing Note (Critical):**
- The `chosen` and `rejected` fields must be **randomly shuffled** to position A and B with equal probability during dataset creation. Without this, the model can learn a positional leak (chosen is always Response A).
- Domain splits: `helpful/` and `harmless/` subdirectories. Use `helpful/` for primary training. `harmless/` reserved for domain-shift evaluation only.

### 2.2 Soft-Label Dataset: Stanford Human Preferences (SHP)

> **This replaces the original crowdsourcing plan entirely.**

| Property | Value |
| :--- | :--- |
| Source | `stanfordnlp/SHP` on Hugging Face |
| Size | ~385,000 Reddit post pairs across 18 domains |
| Format | JSON: `{post, human_ref_A, human_ref_B, score_A, score_B, labels}` |
| Label Type | **Real human vote scores** → compute `p_A = score_A / (score_A + score_B)` |
| IAA Metric | Reddit upvotes as a proxy for collective preference |
| Usage in Project | Soft-label training experiments (RQ3), testing calibration improvement |

**Why SHP over crowdsourcing:**
- Free — zero budget required
- Provides genuine multi-human vote distributions (not binary)
- Available immediately (no IRB, no platform setup, no quality control logistics)
- Enables direct testing of the soft-label hypothesis on Day 1

**Data Quality Gate (SHP):**
Only include SHP pairs where `|score_A - score_B| > 10` AND total votes > 20. This filters out pairs where genuine human disagreement is too high to provide a learning signal.

### 2.3 Augmentation Dataset: UltraFeedback (Kept Separate)

| Property | Value |
| :--- | :--- |
| Source | `openbmb/UltraFeedback` on Hugging Face |
| Size | ~1M GPT-4 feedback scores |
| Label Type | **Synthetic AI-generated** (NOT human) |
| Usage in Project | Separate ablation only: "AI-judge supervised" vs "human-supervised" model. NEVER mixed into primary human-preference training set. |

> [!WARNING]
> UltraFeedback labels are from GPT-4, not humans. All experiments using UltraFeedback must be labelled `[AI-Label]` in your W&B run names to prevent accidental conflation with human preference results.

### 2.4 Dataset Split Summary

| Split | Source | Size | Purpose |
| :--- | :--- | :--- | :--- |
| Train (Hard Label) | HH-RLHF `helpful/` | ~120K | Primary classifier training |
| Train (Soft Label) | SHP (filtered) | ~80K | Soft-label calibration experiment |
| Validation | HH-RLHF `helpful/` | ~20K | Hyperparameter tuning, early stopping |
| Test (In-Domain) | HH-RLHF `helpful/` | ~29K | Accuracy, AUC, Brier Score |
| Test (Bias) | 5,000 pairs (swap + length variants) | 10K augmented | Swap Test, Length Perturbation |
| Test (Domain Shift) | HH-RLHF `harmless/` | ~30K | Generalization gap measurement |

---

## 3. Token Budget Strategy

> **This is the most critical engineering decision and must be finalized before writing any training code.**

All transformer encoders (BERT, DeBERTa) have a hard limit of **512 tokens**. The input format for a cross-encoder is:

```
[CLS] <Prompt> [SEP] <Response A> [SEP] <Response B> [SEP]
```

**Token Allocation (fixed for all experiments):**

| Component | Max Tokens | Truncation Strategy |
| :--- | :---: | :--- |
| `[CLS]` + `[SEP]` × 3 | 4 (reserved) | N/A |
| Prompt | 128 | Truncate from LEFT (remove old context, keep final user turn) |
| Response A | 190 | Truncate from RIGHT |
| Response B | 190 | Truncate from RIGHT |
| **Total** | **512** | |

**Implementation (add to `utils.py`):**
```python
def format_input(prompt: str, resp_a: str, resp_b: str, tokenizer, 
                 max_prompt=128, max_resp=190) -> dict:
    """
    Formats a pairwise preference input with strict token budgeting.
    Truncates prompt from the LEFT to preserve the final user turn.
    """
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_tokens = prompt_tokens[-max_prompt:]  # Keep LAST N tokens
    
    resp_a_tokens = tokenizer.encode(resp_a, add_special_tokens=False)[:max_resp]
    resp_b_tokens = tokenizer.encode(resp_b, add_special_tokens=False)[:max_resp]
    
    input_ids = (
        [tokenizer.cls_token_id] +
        prompt_tokens +
        [tokenizer.sep_token_id] +
        resp_a_tokens +
        [tokenizer.sep_token_id] +
        resp_b_tokens +
        [tokenizer.sep_token_id]
    )
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
```

---

## 4. Model Architecture

### 4.1 Model Cascade

| Tier | Model | Parameters | Training Location | Notes |
| :--- | :--- | :---: | :--- | :--- |
| **T0 — Baseline** | TF-IDF + Logistic Regression | ~10K vocab features | M4 Mac (CPU) | Sanity check baseline |
| **T1 — Sequential** | BiLSTM (2 layers, hidden=256) | ~8M | M4 Mac (MPS) | Non-transformer sequential baseline |
| **T2 — Primary** | BERT-base-uncased Cross-Encoder | 110M | M4 Mac (MPS) / HPC | Dev subset on M4, full run on HPC |
| **T3 — Advanced** | DeBERTa-v3-base Cross-Encoder | 125M | HPC | State-of-the-art for text pair tasks |
| **T4 — Upper Bound** | DeBERTa-v3-large Cross-Encoder | 435M | **HPC only** | Requires 24GB+ VRAM |
| **T5 — Ranker** | Bi-Encoder (shared BERT encoder) | 110M | HPC | Speed vs. accuracy tradeoff study |
| **T6 — LLM Judge** | LLaMA-3-8B (GGUF Q4, few-shot) | 8B | M4 Mac (inference only) | Modern benchmark, NOT fine-tuned |

### 4.2 Calibration Techniques

Applied **after** training to all T2+ models:

| Method | Implementation | Library |
| :--- | :--- | :--- |
| Temperature Scaling | Tune scalar `T` on validation set to minimize NLL | Custom (`calibrate.py`) |
| Isotonic Regression | Non-parametric post-hoc probability calibration | `sklearn.isotonic` |
| MC Dropout | Enable dropout at inference, run 10 forward passes | PyTorch `model.train()` mode during inference |
| Deep Ensemble | Train 3 models with different seeds, average logits | Parallel training on HPC |
| Soft-label Training | Set BCE target = vote fraction from SHP | Custom `SoftBCELoss` in PyTorch |

```python
class SoftBCELoss(nn.Module):
    """Cross-entropy loss that accepts float targets [0, 1] for soft-label training."""
    def forward(self, logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.squeeze(-1))
        return F.binary_cross_entropy(probs, soft_labels.float())
```

---

## 5. Evaluation Design

### 5.1 Primary Metrics

| Metric | Description | Target (DeBERTa-v3-base) |
| :--- | :--- | :--- |
| **Pairwise Accuracy** | % pairs where predicted winner = human majority | > 72% |
| **AUC-ROC** | Area under the curve for preference scoring | > 0.80 |
| **Brier Score** | Mean squared error of predicted probabilities | < 0.20 |
| **ECE** | Expected Calibration Error (15 bins) | < 0.08 (soft-label model) |
| **Swap Consistency** | % pairs where prediction flips when A/B swapped | > 90% (ideally ~100%) |
| **Length Bias Score** | Correlation between length difference and predicted winner | r < 0.15 (post-debiasing) |

### 5.2 Bias Evaluation Suite

**Test A — Swap Test (Positional Bias):**
```python
# For each test pair (prompt, resp_a, resp_b):
# Run model in both orders, check if prediction flips correctly
pred_forward  = model(prompt, resp_a, resp_b)   # Should predict winner = A
pred_swapped  = model(prompt, resp_b, resp_a)   # Should predict winner = B (was A)

# Consistency: prediction should flip when inputs are swapped
consistency = (pred_forward.argmax() != pred_swapped.argmax()).float().mean()
# Target: consistency > 0.90
```

**Test B — Length Perturbation:**
- Generate 500 pairs where Response A is a **compressed version** of Response B (same information, 40% fewer words, using sentence compression).
- A biased model will favor the longer Response B despite equal content quality.
- Measure: `Accuracy on length-perturbed pairs` (target > 65% — non-trivial baseline is 50%).

**Test C — Domain Shift:**
- Train on `HH-RLHF/helpful/` → Test on `HH-RLHF/harmless/`
- Report accuracy drop as the **Domain Shift Gap metric**.

**Test D — Calibration on Disagreement Pairs:**
- Filter SHP test set for high-disagreement pairs: `0.4 < p_A < 0.6`
- A well-calibrated model should output probabilities near 0.5 for these pairs.
- Measure ECE on this subset specifically.

### 5.3 Inter-Annotator Agreement (IAA) — Data Quality Gate

Before including any pair in soft-label training (from SHP):
- Compute `Kappa` approximation using vote ratio: `kappa_proxy = 2 * |p_A - 0.5|`
- Pairs with `kappa_proxy < 0.2` (i.e., very close to 50/50 with few votes) are **excluded** from training.
- All IAA statistics reported in the EDA notebook.

---

## 6. Experiment Tracking — Weights & Biases (W&B)

**Set up on Day 1 of Week 1. Non-negotiable.**

```bash
pip install wandb
wandb login   # Use free academic account
```

**Naming Convention for Runs:**

```
{model_tier}_{label_type}_{dataset}_{seed}
# Examples:
# t2_bert_hard_hh_seed42
# t3_deberta_soft_shp_seed0
# t6_llama_fewshot_hh_inference
```

**Logged Automatically:**
- Training loss + validation loss per step
- Accuracy, AUC, ECE, Brier Score per epoch
- Swap Test consistency score
- Length bias correlation `r`
- GPU memory usage and training time per step
- Model checkpoint path

```python
import wandb

wandb.init(
    project="pairwise-preference-model",
    name=f"t3_deberta_soft_shp_seed{args.seed}",
    config={
        "model": args.model_name,
        "label_type": args.label_type,   # "hard" or "soft"
        "dataset": args.dataset,          # "hh_rlhf" or "shp"
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_seq_len": 512,
        "seed": args.seed,
    }
)
```

---

## 7. Code Repository Structure

```
preference-model-project/
│
├── data/
│   ├── raw/                    # Downloaded datasets (HH-RLHF, SHP) — gitignored
│   ├── processed/              # Tokenized, split, shuffled train/val/test splits
│   │   ├── hh_rlhf_hard/       # Hard-label splits
│   │   └── shp_soft/           # Soft-label splits (with vote fractions)
│   └── bias_test/              # Length-perturbed pairs, swap-test pairs
│
├── src/
│   ├── data/
│   │   ├── download.py         # HuggingFace dataset download scripts
│   │   ├── preprocess.py       # JSONL → pandas pipeline, shuffle, split
│   │   └── dataset.py          # PyTorch Dataset classes (hard & soft label)
│   ├── models/
│   │   ├── tfidf_baseline.py   # TF-IDF + Logistic Regression
│   │   ├── bilstm.py           # BiLSTM encoder + classifier
│   │   ├── cross_encoder.py    # BERT/DeBERTa cross-encoder wrapper
│   │   └── bi_encoder.py       # Bi-encoder (shared encoder) wrapper
│   ├── training/
│   │   ├── train.py            # Main training loop (HuggingFace Trainer)
│   │   ├── losses.py           # SoftBCELoss, pairwise hinge loss
│   │   └── hpc_launch.sh       # SLURM job script for IIT Bhilai cluster
│   ├── evaluation/
│   │   ├── metrics.py          # Accuracy, AUC, ECE, Brier Score
│   │   ├── bias_tests.py       # Swap Test, Length Perturbation Test
│   │   └── domain_shift.py     # Cross-domain generalization evaluation
│   ├── calibration/
│   │   ├── temperature_scaling.py
│   │   ├── isotonic.py
│   │   └── mc_dropout.py
│   └── utils/
│       ├── tokenization.py     # Token budget formatter (format_input fn)
│       ├── logging.py          # W&B integration helpers
│       └── seed.py             # Global seed setter for reproducibility
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Dataset stats, IAA analysis, length distributions
│   ├── 02_Baselines.ipynb      # TF-IDF and BiLSTM results
│   ├── 03_TransformerResults.ipynb  # BERT/DeBERTa accuracy & calibration plots
│   ├── 04_BiasAnalysis.ipynb   # Swap Test, Length Bias visualizations
│   └── 05_CalibrationCurves.ipynb  # Reliability diagrams, ECE comparisons
│
├── demo/
│   ├── app.py                  # Streamlit demo (local deployment)
│   └── api.py                  # FastAPI endpoint (optional, Week 6)
│
├── configs/
│   ├── bert_base.yaml          # Hyperparameters for BERT-base run
│   ├── deberta_base.yaml       # Hyperparameters for DeBERTa-v3-base
│   └── deberta_large.yaml      # Hyperparameters for HPC run
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 8. Training Configuration

### 8.1 Local Mac M4 Config (Development & Prototyping)

```yaml
# configs/bert_base_local.yaml
model_name: "bert-base-uncased"
device: "mps"                      # Apple Metal GPU
dataset: "hh_rlhf"
subset_size: 20000                  # Dev subset for rapid iteration
batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 2e-5
num_epochs: 3
max_seq_length: 512
fp16: true                          # Halves memory usage
warmup_ratio: 0.1
early_stopping_patience: 2
label_type: "hard"                  # "hard" or "soft"
seed: 42
wandb_project: "pairwise-preference-model"
```

### 8.2 HPC Cluster Config (Full-Scale Training)

```bash
#!/bin/bash
# hpc_launch.sh — SLURM script for IIT Bhilai GPU cluster
#SBATCH --job-name=pref-deberta-large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_deberta_large.out

module load python/3.10 cuda/11.8
source venv/bin/activate

python src/training/train.py \
  --config configs/deberta_large.yaml \
  --device cuda \
  --gradient_checkpointing true \   # CRITICAL for 435M model memory management
  --fp16 true \
  --seed 42
```

**HPC Memory Optimization (DeBERTa-v3-large):**
```python
# In TrainingArguments for large model runs
training_args = TrainingArguments(
    gradient_checkpointing=True,    # Halves activation memory at 20% speed cost
    fp16=True,                      # Halves weight/gradient memory
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch size = 32
    dataloader_num_workers=4,
    report_to="wandb",
)
```

---

## 9. Revised 6-Week Timeline

> **Core Change:** Crowdsourcing removed entirely. All annotation work replaced by SHP dataset download and processing (< 30 minutes).

```
WEEK 1 — Data & Infrastructure Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 1:  Environment setup (venv, requirements.txt, W&B account, HPC access request)
Day 2:  Download HH-RLHF + SHP, run EDA notebook (01_EDA.ipynb)
        → Measure token length distributions, IAA proxy scores, label balance
Day 3:  Implement preprocess.py: shuffling, splitting, token budget formatter
Day 4:  Implement TF-IDF + LR baseline → run on full dataset (CPU, ~2 min)
Day 5:  Implement BiLSTM → train on 20k dev subset on M4 (~15 min)
        → Log all results to W&B

WEEK 2 — Transformer Baselines
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 1:  Implement cross_encoder.py (BERT-base wrapper)
Day 2:  Fine-tune BERT-base on 20k dev subset (M4, ~20 min), debug training loop
Day 3:  Run BERT-base on full 169k HH-RLHF (M4 overnight, ~3 hrs)
Day 4:  Implement evaluation metrics (metrics.py): Accuracy, AUC, ECE, Brier
Day 5:  Write 02_Baselines.ipynb with results table and plots
        → Confirm W&B dashboard shows all 4 models (T0–T2) with consistent metrics

WEEK 3 — Soft Labels & DeBERTa-base
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 1:  Implement SoftBCELoss, soft-label Dataset class, SHP preprocessing
Day 2:  Fine-tune BERT-base with soft labels on SHP (M4, ~2 hrs)
        Compare ECE: hard-label model vs soft-label model → first test of RQ3
Day 3:  [HPC REQUEST SUBMITTED] Launch DeBERTa-v3-base on full HH-RLHF (HPC)
Day 4:  Implement bias_tests.py: Swap Test + Length Perturbation Test
Day 5:  Run bias tests on all trained models, document results in W&B

WEEK 4 — Advanced Models & Bias Experiments
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 1:  [HPC] DeBERTa-v3-large training (8 hrs on A100/V100)
        [M4]  LLaMA-3-8B GGUF Q4 inference on test set (RQ1 LLM judge baseline)
Day 2:  Implement bi_encoder.py, train bi-encoder on HPC (speed/accuracy tradeoff)
Day 3:  Run full bias test suite on DeBERTa models: Swap, Length, Domain Shift
Day 4:  Implement mc_dropout.py, temperature_scaling.py, isotonic.py
Day 5:  Write 04_BiasAnalysis.ipynb — visualize positional/verbosity bias magnitude

WEEK 5 — Calibration Experiments & Ablations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 1:  Apply Temperature Scaling to all T2–T4 models, measure ECE before/after
Day 2:  Train deep ensemble (3 seeds × DeBERTa-v3-base on HPC)
        Compare ensemble uncertainty vs MC Dropout uncertainty
Day 3:  Ablation 1: Remove prompt from input (response-only model) — does context matter?
Day 4:  Ablation 2: Train UltraFeedback model (AI labels) → compare to human-label model
Day 5:  Write 05_CalibrationCurves.ipynb with reliability diagrams for all models

WEEK 6 — Demo, Documentation & Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Day 1:  Build Streamlit demo (demo/app.py): input prompt + 2 responses → see winner, confidence, bias flags
Day 2:  Finalize README.md, add reproducibility instructions and seed list
Day 3:  Write project report (Introduction, Methods, Results, Discussion, Conclusion)
Day 4:  Create final results table comparing all 7 model tiers across all metrics
Day 5:  Code review, repo cleanup, push to GitHub, record short demo video
```

---

## 10. Compute Assignment Summary

| Task | Hardware | Est. Time |
| :--- | :--- | :--- |
| Data download, preprocessing | M4 Mac (CPU) | 30 min |
| EDA Notebook, token analysis | M4 Mac (CPU/MPS) | 1 hr |
| TF-IDF + BiLSTM training | M4 Mac (CPU) | 20 min total |
| BERT-base dev subset (20k) | M4 Mac (MPS) | 20 min |
| BERT-base full run (169k) | M4 Mac (MPS, overnight) | 3–3.5 hrs |
| Soft-label BERT-base on SHP | M4 Mac (MPS) | 2–2.5 hrs |
| LLaMA-3-8B inference (test set) | M4 Mac (MPS, GGUF Q4) | 45–60 min |
| DeBERTa-v3-base full run | **IIT Bhilai HPC** | ~4 hrs |
| DeBERTa-v3-large full run | **IIT Bhilai HPC** | ~8 hrs |
| Deep Ensemble (3× DeBERTa-base) | **IIT Bhilai HPC** | ~12 hrs |
| Bias tests (inference only) | M4 Mac (MPS) | 30 min |
| Calibration experiments | M4 Mac (CPU) | 2 hrs |
| Streamlit demo | M4 Mac | 1 day |

**Total M4 Mac compute:** ~12–15 hours spread across 6 weeks (all manageable)  
**Total HPC compute:** ~24 hours of GPU time (request a 32-hour slot)

---

## 11. Demo: Streamlit Application

```python
# demo/app.py
import streamlit as st
import torch
from src.models.cross_encoder import CrossEncoderModel
from src.utils.tokenization import format_input
from src.calibration.temperature_scaling import TemperatureScaler

st.set_page_config(page_title="Pairwise Preference Model", layout="wide")
st.title("🔍 Pairwise LLM Response Preference Predictor")
st.markdown("*Predicts which response a human would prefer, with calibrated confidence and bias checks.*")

with st.form("preference_form"):
    prompt = st.text_area("📝 Prompt", placeholder="Enter the question or prompt here...")
    col1, col2 = st.columns(2)
    with col1:
        resp_a = st.text_area("🅰️ Response A", placeholder="Enter Response A...", height=200)
    with col2:
        resp_b = st.text_area("🅱️ Response B", placeholder="Enter Response B...", height=200)
    submitted = st.form_submit_button("Predict Winner")

if submitted and prompt and resp_a and resp_b:
    # Forward prediction
    p_a_forward = model.predict(prompt, resp_a, resp_b)
    # Swapped prediction (bias check)
    p_a_swapped = model.predict(prompt, resp_b, resp_a)
    positional_bias = abs((p_a_forward) - (1 - p_a_swapped))
    
    winner = "Response A 🅰️" if p_a_forward > 0.5 else "Response B 🅱️"
    confidence = max(p_a_forward, 1 - p_a_forward)
    
    st.subheader(f"Predicted Winner: **{winner}**")
    st.metric("Confidence", f"{confidence:.1%}")
    st.metric("Positional Bias Score", f"{positional_bias:.3f}", 
              delta="Low ✅" if positional_bias < 0.05 else "High ⚠️",
              delta_color="normal")
    
    if confidence < 0.60:
        st.warning("⚠️ Low confidence — human review recommended for this pair.")
    if positional_bias > 0.10:
        st.error("🔴 High positional bias detected — model prediction may depend on order, not content.")
    
    st.progress(p_a_forward, text=f"P(A preferred) = {p_a_forward:.3f}")
```

---

## 12. Risk Management

| Risk | Likelihood | Impact | Mitigation |
| :--- | :---: | :---: | :--- |
| HPC cluster not available in Week 3 | Medium | High | Fallback: Kaggle free T4 GPU (20 hrs/week) or Google Colab A100 |
| DeBERTa-v3-large OOM on HPC | Low | Medium | Enable `gradient_checkpointing=True`, reduce batch to 4 |
| M4 MPS instability with DeBERTa attention | Low | Medium | Use `torch.float32` instead of `fp16` if MPS errors occur |
| SHP soft labels too noisy for training | Low | Medium | Apply IAA quality gate (exclude pairs with vote gap < 10) |
| 6-week timeline slippage | Medium | Medium | Weeks 1–2 are deliberately lighter; buffer is built into Week 5 |

---

## 13. Prioritized Reading List (Updated)

| Paper | Why Critical |
| :--- | :--- |
| Lambert (2026) — *RLHF Survey* | Comprehensive background on reward models in alignment |
| Stiennon et al. (2022) — *HH-RLHF* | Describes the primary dataset; understand the label schema |
| Rafailov et al. (2023) — *DPO* | Pairwise BCE loss formulation — directly informs our loss function |
| Liu et al. (2024) — *Pairwise Preference Biases* | Positional and verbosity bias analysis — directly cited in bias evaluation |
| Cui et al. (2023) — *UltraFeedback* | AI-generated feedback dataset — used only in ablation |
| **Pezeshkpour & Hruschka (2023)** — *Large Language Models Sensitivity* | NEW: Specifically measures positional bias in LLM judges — directly supports RQ2 |
| **Guo et al. (2017)** — *Calibration of Modern Neural Networks* | Temperature scaling foundations for ECE measurement |

---

## 14. Publication Pathway

The strongest contribution of this project — if executed rigorously — is publishable at:

| Venue | Target | Reasoning |
| :--- | :---: | :--- |
| **EMNLP 2026 Findings** | ⭐⭐⭐ | Evaluation-focused NLP work with bias analysis fits perfectly |
| **ACL ARR Rolling** | ⭐⭐⭐ | Rolling submission, can submit after 6-week project completes |
| **Workshop on Trustworthy NLP (TrustNLP @ ACL)** | ⭐⭐⭐ | Reliability and calibration are the core workshop themes |
| **NeurIPS Alignment Workshop** | ⭐⭐ | Broader alignment audience, but evaluation bias is relevant |

**The paper's core claim to novelty:**  
> *"We provide the first systematic, head-to-head study of positional bias, verbosity bias, and calibration failure across a cascade of preference models ranging from TF-IDF to DeBERTa-v3-large, using both hard and soft human preference labels."*

---

*Document Version: 2.0 | Last Updated: October 2026 | IIT Bhilai NLP Research Group*
