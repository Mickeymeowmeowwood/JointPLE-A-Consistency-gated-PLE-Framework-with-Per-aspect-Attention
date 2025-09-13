# JointPLE-A-Consistency-gated-PLE-Framework-with-Per-aspect-Attention
JointPLE: A Consistency-gated PLE Framework with Per-aspect Attention for Chinese Aspect-level Sentiment Classification and Sentence-level Rating Prediction
# JointPLE: Chinese ACSA + Rating Prediction

This repository contains the code, data artifacts, and scripts used in the paper:

> *JointPLE: A Consistency-gated PLE Framework with Per-aspect Attention for Chinese Aspect-level Sentiment Classification and Sentence-level Rating Prediction*

---

## Table of Contents

- [Overview](#overview)  
- [Contents](#contents)  
- [Getting Started](#getting-started)  
- [Reproducing Experiments](#reproducing-experiments)  
- [Figures](#figures)  
- [Data Availability & Artifacts](#data-availability--artifacts)  
- [Environment & Dependencies](#environment--dependencies)  
- [Results & Benchmarks](#results--benchmarks)  
- [License](#license)  

---

## Overview

JointPLE is a multi-task model combining Progressive Layered Extraction (PLE) with a consistency gate and per-aspect attention, trained on the ASAP Chinese dining dataset for both Aspect-Category Sentiment Analysis (ACSA) and Rating Prediction (RP). This repository provides all necessary components to reproduce the main results, tables, and figures in the paper.

---

## Contents

Here is a summary of what is included:

| Directory / File | Description |
|------------------|-------------|
| `src/ple/` | Main model implementation (JointPLE) |
| `src/ple11/` | Ablation version code excluding certain components (“ple11”) |
| `src/hpo/` | Hyperparameter search (grid search) scripts and configuration files |
| `scripts/train_ple.sh` | Script to train the JointPLE main model |
| `scripts/train_ple11.sh` | Script to train the ablation version ple11 |
| `scripts/run_hpo.sh` | Script to run grid search over hyperparameters |
| `scripts/plot/figure1.py` … `figure11.py` | Scripts to generate Figures 1–11 in the paper |
| `scripts/plots_make_all.sh` | Master script to run all figure scripts and output to `figs/` |
| `results/` | Result files: `ple_main_model.csv`, `ablation_summary.csv`, `paired_tests.csv`, `grid20_summary.json` |
| `figs/` | Final PDF/SVG outputs of Figures 1–11 for visual comparison |
| `README.md` | This document |
| `LICENSE` | License file for code and artifacts |
| `environment.yml` or `requirements.txt` | Libraries, versions, and environment configuration |

---

## Getting Started

To get up and running:

1. Clone this repository:  
   ```bash
   git clone [https://github.com/<your-org>/<your-repo>](https://github.com/Mickeymeowmeowwood/JointPLE-A-Consistency-gated-PLE-Framework-with-Per-aspect-Attention.git
   cd JointPLE-A-Consistency-gated-PLE-Framework-with-Per-aspect-Attention
Install dependencies:

bash
复制代码
conda env create -f environment.yml
# or
pip install -r requirements.txt
Download the ASAP dataset following its official instructions. (Do not include raw ASAP data in this repo if license prohibits redistribution.)

Run main experiments or reproduce results:

Train main model: bash scripts/train_ple.sh

Train ablation: bash scripts/train_ple11.sh

Run hyperparameter search: bash scripts/run_hpo.sh

Generate figures:

bash
复制代码
bash scripts/plots_make_all.sh
Figures
Each figure in the paper corresponds to one plotting script. The PDF/SVG images are provided for direct visual comparison:

Fig.1 → figs/S2_dataset_profile.pdf

Fig.2 → figs/fig2a_hparam_heatmap.pdf, figs/fig2b_top_runs.pdf

Fig.3 → figs/fig3a_ablation_mae.pdf, figs/fig3b_ablation_rpacc.pdf, figs/fig3c_signif_heatmap.pdf

Fig.4 → figs/Fig4_seedvar_test_acc.pdf, figs/Fig4_seedvar_test_f1.pdf, figs/Fig4_seedvar_test_mae.pdf, figs/Fig4_seedvar_test_rp_acc.pdf

Fig.5 → figs/rp_calib_calibration.pdf, figs/rp_calib_error_hist.pdf

Fig.6 → figs/acsa_cm.pdf, figs/acsa_cm_counts.pdf

Fig.7 → figs/attn_case_fix_attention.pdf

Fig.8 → figs/fig8_sensitivity.pdf

Fig.9 → figs/gates_acsa.pdf, figs/gates_rp.pdf, figs/gates_share.pdf

Fig.10 → figs/S3_efficiency.pdf

Fig.11 → figs/S4_embed.pdf

Data Availability & Artifacts

The ASAP dataset is publicly available under its official terms.

All derived artifacts supporting the results (including result CSV, JSON files, and figure scripts) are deposited here.

Result files include: ple_main_model.csv, ablation_summary.csv, paired_tests.csv, grid20_summary.json.

These artifacts provide the minimal dataset necessary to interpret, replicate, and build upon the analyses in the paper.

If any result or data is not publicly shareable for licensing or privacy reasons, contact the authors to request access.

Environment & Dependencies
Python version: 3.12.3 (e.g. 3.8+)

Key libraries and versions (for example):

PyTorch ≥ 1.8

Transformers (HuggingFace) ≥ 4.41.1

NumPy, SciPy, scikit-learn (for metrics, bootstrap)

Other usual dependencies: tqdm, matplotlib, etc.

Hardware: GPU(s) with at least ~X GB memory (for ASAP’s long reviews)

Reproducibility:

Fixed random seeds: {42, 47, 52, 57, 62}

All training / evaluation scripts log hyperparameters and configuration files.

Results & Benchmarks
Results in this repository match those reported in the paper. For example:

Metric	Baseline	JointPLE (main)
RP MAE	0.4266	0.4167
ACSA Macro-F1	…	…

See results/ple_main_model.csv and results/ablation_summary.csv for full details.

License
This work is licensed under Apache-2.0. See the LICENSE file for details.

References
ASAP dataset: Bu et al. (2021), NAACL-HLT
