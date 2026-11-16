# Reliable Pairwise Preference Modeling for LLM Responses
### An Empirical Investigation of Positional Bias, Verbosity Bias, and Calibration in RLHF Reward Models

---

## Executive Summary & Research Questions

In Reinforcement Learning from Human Feedback (RLHF), reward models are traditionally ranked almost exclusively by **pairwise accuracy** on benchmark splits. However, accuracy alone treats every prediction as a scalar binary hit or miss, completely concealing critical failure modes.

This project audits a 7-tier cascade of preference models (from a 1-parameter length rule to an 8-Billion parameter LLaMA-3 judge) across Anthropic HH-RLHF (111K train, 6.2K test pairs) and Stanford Human Preferences (SHP, 72K train, 18.4K test pairs), answering three foundational Research Questions:

1. **RQ1 (Model Scale vs. Domain Fine-Tuning):** Can a small fine-tuned encoder (184M parameters) predict human preferences as accurately and reliably as an 8B prompted LLM judge?
   - **Finding:** A fine-tuned DeBERTa-v3 ensemble outperforms LLaMA-3-8B-Instruct by **+4.7 percentage points** (66.5% vs. 61.9%, $p < 0.001$) and exhibits an order-of-magnitude lower calibration error ($\text{ECE} = 0.039$ vs. $0.313$).
2. **RQ2 (Non-Content Bias Sensitivity):** How sensitive are preference models to response order (positional bias) and length (verbosity bias)?
   - **Finding:** Cross-encoders flip their decision on **14% to 23% of pairs** when responses are swapped. Training-time swap augmentation fails to eliminate this ($p = 0.10$), whereas **inference-time symmetrization** ($p_{\text{sym}} = \frac{1}{2}(p_{AB} + 1 - p_{BA})$) restores 99.9% consistency and gives a $+1.0\%$ accuracy boost ($p < 0.05$). In our 500-pair compression study (Test B), models systematically favor the longer version of identical-content answers 59% to 68% of the time.
3. **RQ3 (Soft Distributional Supervision):** Does training on real multi-annotator vote shares from SHP improve calibration on contentious pairs?
   - **Finding:** Soft-label training cuts ECE by **$3\times$ to $4.5\times$** (down to $0.024$) and reduces KL divergence from human vote shares by **36% to 60%** across all architectures.

---

## Project Repository Map

```
NLP_Preference_Model/
├── README.md                      <- Master project guidebook & exploration directions (this file)
├── requirements.txt               <- Core dependencies
├── requirements-lock.txt          <- Exact frozen environment versions
├── notebooks/                     <- Interactive Jupyter & Python Notebooks
│   ├── 01_EDA_and_Preprocessing.ipynb  <- Visual EDA, token budget profiling, and quality gating
│   ├── 02_Baselines.ipynb              <- T0 TF-IDF, length heuristics, and OpenAssistant eval
│   ├── 03_Train_DeBERTa_Hard.ipynb     <- DeBERTa-v3 fine-tuning with live tqdm progress bars
│   ├── 04_Master_Results_and_Figures.ipynb <- Master visualization of all 36 runs and research questions
│   ├── 01_eda.py                       <- Batch script for generating EDA statistics
│   └── 02_results.py                   <- Script to compile results tables and render paper plots
│
├── report/                        <- Complete Submission Course Report & Paper
│   ├── report.pdf                 <- Compiled camera-ready PDF report
│   └── figures/                   <- High-res vector PDF & raster figures
│
├── src/                           <- Modular, clean, production-grade source code
│   ├── data/                      <- Dataset loading, streaming, and preprocessing
│   │   ├── dataset.py             <- PairDataset, CrossCollator, SiameseCollator, load_split
│   │   ├── download.py            <- HuggingFace streaming download CLI
│   │   └── preprocess.py          <- Dialogue parsing, A/B coin-flip, IAA gating
│   ├── models/                    <- Model cascade architectures (T0 to T6)
│   │   ├── cross_encoder.py       <- BERT & DeBERTa-v3 pairwise cross-encoders
│   │   ├── bi_encoder.py          <- Siamese bi-encoder (100% order-invariant by construction)
│   │   ├── bilstm.py              <- 2-layer BiLSTM baseline (Siamese & Cross modes)
│   │   ├── tfidf_baseline.py      <- Anti-symmetric TF-IDF feature-difference model
│   │   ├── length_baseline.py     <- Pure log-length ratio logistic regression
│   │   └── llm_judge.py           <- Few-shot LLaMA-3-8B-Instruct evaluator
│   ├── training/                  <- Training loops and loss formulations
│   │   ├── train.py               <- Main training entrypoint with YAML overrides
│   │   ├── trainer.py             <- Training loop with early stopping on validation loss
│   │   └── losses.py              <- SoftBCELoss (continuous float target support)
│   ├── evaluation/                <- Metric computation & bias testing suite
│   │   ├── metrics.py             <- Accuracy, AUC, Brier Score, and 15-bin ECE
│   │   ├── bias_tests.py          <- The Swap Test & Qwen 40% compression study (Test B)
│   │   ├── evaluate.py            <- Shared evaluation runner across all splits
│   │   └── significance.py        <- Paired-bootstrap tests (2,000 resamples) with p-values
│   ├── calibration/               <- Post-hoc uncertainty & calibration methods
│   │   ├── temperature_scaling.py <- Vectorized temperature scaling via NLL minimization
│   │   ├── mc_dropout.py          <- 10-pass Monte Carlo Dropout inference
│   │   ├── ensemble.py            <- 3-seed Deep Ensemble logit averaging
│   │   ├── isotonic.py            <- Isotonic regression calibration
│   │   └── calibrate.py           <- Batch calibration runner
│   └── utils/                     <- Token budget helpers & determinism
│       ├── tokenization.py        <- Strict 128 / 190 / 190 token budget formatter
│       ├── seed.py                <- Global deterministic seed setter
│       └── tracking.py            <- Lightweight logging & MLflow experiment tracker
│
├── results/                       <- All 36+ precomputed experimental runs & logs
│   ├── tables/                    <- Markdown & CSV tables (Tables 1 through 8)
│   │   ├── table1_hh.md           <- Master benchmark on Anthropic HH-RLHF helpful
│   │   ├── table2_shp_soft_vs_hard.md <- Soft vs. Hard label calibration comparison
│   │   ├── table5_test_b_length.md    <- Test B length perturbation results
│   │   ├── table6_uncertainty.md      <- Model uncertainty vs. human disagreement
│   │   ├── table7_significance.md     <- Statistical bootstrap significance tests
│   │   └── table8_transfer.md         <- Cross-domain helpful vs. harmless transfer
│   ├── significance.csv           <- Exact paired-bootstrap p-values and confidence intervals
│   └── t*/                        <- Per-run metric JSONs and calibration files
│
├── configs/                       <- Modular hyperparameters (YAML)
│   ├── bert_base.yaml             <- BERT-base cross-encoder config
│   ├── deberta_base.yaml          <- DeBERTa-v3-base config (fp32, batch 16, grad accum 2)
│   ├── bert_bi.yaml               <- Siamese bi-encoder config
│   └── deberta_large.yaml         <- Large-scale configuration
│
└── scripts/                       <- Automation & queue execution scripts
    ├── run_week2.sh               <- Baselines & initial BERT pipeline
    ├── run_week3.sh               <- Soft-label DeBERTa on SHP
    ├── run_week4.sh               <- Bi-encoders, LLM judge, and ensembles
    ├── run_week5.sh               <- UltraFeedback ablation & harmless transfer
    └── status.sh                  <- Shell status monitor for running jobs
```

---

## Step-by-Step Directions to Explore the Codebase

You can explore the project through four different pathways:

### Pathway 1: Explore Precomputed Results & Scientific Findings (Fastest, No GPU Needed)
All 36 experiment configurations are pre-calculated with exact bootstrap confidence intervals and statistical tests.
- **Inspect Master Benchmark (Table 1):** Open [`results/tables/table1_hh.md`](results/tables/table1_hh.md).
- **Inspect Soft vs. Hard Calibration (Table 2):** Open [`results/tables/table2_shp_soft_vs_hard.md`](results/tables/table2_shp_soft_vs_hard.md).
- **Inspect Length Compression Test (Table 5):** Open [`results/tables/table5_test_b_length.md`](results/tables/table5_test_b_length.md).
- **Inspect Statistical Significance Tests:** Open [`results/tables/table7_significance.md`](results/tables/table7_significance.md) or [`results/significance.csv`](results/significance.csv).

### Pathway 2: Interactive Exploration via Jupyter Notebooks
If you want to walk through the project interactively cell-by-cell:
1. **Notebook 01 (`notebooks/01_EDA_and_Preprocessing.ipynb`):**
   - Ingests Anthropic HH-RLHF and Stanford SHP.
   - Demonstrates the A/B coin-flip positional debiasing.
   - Plots token length distributions and validates that the 512-token budget retains $>90\%$ of text without truncation.
2. **Notebook 02 (`notebooks/02_Baselines.ipynb`):**
   - Runs the null hypothesis random baseline and verbosity heuristic.
   - Fits the anti-symmetric TF-IDF + Logistic Regression baseline in under 2 minutes.
   - Proves 100% swap-consistency mathematically and plots reliability diagrams.
3. **Notebook 03 (`notebooks/03_Train_DeBERTa_Hard.ipynb`):**
   - Demonstrates the fine-tuning workflow of `microsoft/deberta-v3-base` on HH-RLHF.
   - Includes live `tqdm` countdown bars, gradient accumulation, and swap-test validation.


### Pathway 3: End-to-End CLI Reproduction
To re-run any individual experiment from the command line:

```bash
# 1. Train BERT-base cross-encoder
python -m src.training.train --config configs/bert_base.yaml

# 2. Train DeBERTa-v3 with soft labels on Stanford Human Preferences
python -m src.training.train --config configs/deberta_base.yaml dataset=shp label_type=soft

# 3. Train Siamese Bi-Encoder (order-invariant by construction)
python -m src.training.train --config configs/bert_bi.yaml

# 4. Run the Swap Test and Length Perturbation Test (Test B)
python -m src.evaluation.bias_tests evaluate t3_deberta_hard_hh_seed42

# 5. Fit Temperature Scaling on validation logits
python -m src.calibration.calibrate --all

# 6. Re-generate all paper figures and tables
python notebooks/02_results.py
```

---

## Master Results Summary Table

The table below summarizes the headline findings evaluated on 6,194 Anthropic HH-RLHF test pairs:

| Tier | Architecture | Parameters | Raw Acc (%) | Symmetrized Acc (%) | AUC-ROC | ECE | Swap Consistency | Harmless Acc |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T0** | Length Only Heuristic | 1 | 55.6% | 55.6% | 0.590 | 0.020 | 0.981 | 42.2% |
| **T0** | TF-IDF + Logistic Regression | 10K | 60.3% | 60.3% | 0.656 | 0.016 | 0.996 | 38.5% |
| **T1** | BiLSTM (Cross-Encoder) | 11M | 61.7% | 62.2% | 0.676 | 0.024 | 0.776 | 37.3% |
| **T1** | BiLSTM (Siamese Bi-Encoder) | 11M | 63.0% | 63.0% | 0.686 | 0.012 | 0.999 | 36.1% |
| **T2** | BERT-base Cross-Encoder | 110M | 63.6% | 64.6% | 0.706 | 0.022 | 0.773 | 35.2% |
| **T2** | BERT-base (3-Seed Ensemble) | 330M | 64.4% | 65.1% | 0.714 | 0.024 | 0.815 | 34.6% |
| **T5** | BERT-base Bi-Encoder | 110M | 64.6% | 64.6% | 0.711 | 0.023 | 0.997 | 34.3% |
| **T3** | DeBERTa-v3-base Cross-Encoder | 184M | 65.8% | 66.8% | 0.726 | 0.064 | 0.817 | 34.6% |
| **T3** | **DeBERTa-v3 (3-Seed Ensemble)** | 552M | **66.5%** | **67.5%** | **0.736** | **0.039** | **0.862** | 33.8% |
| **T6** | LLaMA-3-8B-Instruct (3-Shot) | 8B | 61.9% | 63.6% | 0.670 | 0.313 | 0.601 | **52.6%** |

*All differences between DeBERTa-v3 and LLaMA-3-8B are statistically significant at $p < 0.001$ using a paired bootstrap test with 2,000 resamples.*

---
