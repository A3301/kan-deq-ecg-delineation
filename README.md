# kan-deq-ecg-delineation

Code, analysis protocols and derived results for the manuscript

> A. Akdağlı, H. Yanık, A. Yıldız. *Accuracy and Computational Cost of Kolmogorov–Arnold and Deep Equilibrium Networks for ECG Delineation: An Equal-Protocol Benchmark.* Manuscript submitted for publication.

The study compares eight single-lead ECG delineators under one training protocol:
- conventional baselines: U-Net, CNN–BiLSTM and a patch Transformer;
- the proposed models: U-Net-KAN, ConvKAN and a deep equilibrium model (DEQ);
- controls that isolate the defining component of each proposed model: a parameter-matched GELU head, weight-tied twins unrolled for 8 and 18 steps, and an explicit stack.

Models are trained on LUDB and tested on QTDB, BUT PDB, the MIT-BIH P-wave annotations, Zhejiang, ISP, RDB and the PTB Diagnostic ECGs of the Berlin QT reference.

Trained weights, record-level outputs and the raw-waveform subsets used by individual analyses are too large for GitHub. They are archived on Zenodo; see **Large files** below.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Core library: data loading and pre-processing (`data.py`, `annotations.py`), subject-level splits (`splits.py`), the eight architectures (`models.py`), training (`train.py`), evaluation and event matching (`evaluate.py`, `metrics.py`, `postprocess.py`) |
| `configs/` | Learning rates selected on fold 0 (`chosen_lr.json`) and the subject-level LUDB split (`splits.json`) |
| `scripts/benchmark/` | Primary benchmark: cross-validation re-evaluation, internal and external evaluation (QTDB, BUT PDB, MIT-BIH P-wave, Zhejiang, ISP), the wavelet reference, head comparison, KAN interpretation and ablations, DEQ solver analyses, latency and memory measurements, tables and figures |
| `analyses/` | One self-contained folder per additional analysis, each with its protocol, scripts, verification code and derived results (see the table below) |
| `analyses/post_hoc/` | The two post hoc analyses reported in the paper (high-frequency proxy versus QRS width; DEQ accuracy by solver convergence on RDB) |
| `figures/figure1/` | Data extraction and drawing script for Figure 1 |
| `docs/` | The dated analysis plan (original, in Turkish) with its SHA-256 digest, and data-source notes |

### Analyses

| Folder | Paper section |
|---|---|
| `hypothesis_statistics`, `hypothesis_waveforms` | Prespecified hypotheses H1–H3, BUT PDB primary endpoint |
| `ECG_Training_Variability_Replay` | Three training runs of all eight architectures (Supplementary Section S9) |
| `BUT_PDB_seed_replay`, `BUT_PDB_uniform_seed_addendum` | BUT PDB P-peak results across training runs (Table S9c) |
| `Interpretability_Replay`, `targeted_replay` | KAN head channel interventions and matched-head control (Section 3.2, Table S2); latency and training-memory profiles (Table S6) |
| `RDB_external_validation`, `RDB_DEQ_external_validation`, `RDB_QRS_duration_audit` | Transport to RDB lead II, the DEQ comparison and QRS duration (Sections 3.5, 3.7; S5, S8) |
| `Reverse_source_training`, `ISP_Common_Test`, `Multilead_LUDB_validation`, `Disjoint_lead_QRS_validation` | Training on RDB and testing on LUDB and ISP; lead-specific and global QRS duration (Sections 3.6, 3.7; S7, S10) |
| `Berlin_PTB_QT_external` | QT agreement on the PTB Diagnostic ECGs (Section 3.7; S11) |
| `QTDB_Observer_Agreement` | Reader agreement on QRS duration in QTDB (S12) |
| `Interval_agreement_figure` | Figure 8 |
| `PTBXL_coordinate_audit` | Pre-inference audit of a candidate annotation release that contributed no results |

Each folder has its own `README.txt`, `REPRODUCE.txt` or `protocol.txt` with the exact commands, and a `SHA256SUMS.txt`. Entries for files that are not in this repository refer to the Zenodo archive.

## Installation

Python 3.12 on CPU; no GPU is required for inference or the statistical analyses.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The library modules are imported from `src/`. Run the benchmark scripts with `src` on the path, for example:

```bash
PYTHONPATH=src python scripts/benchmark/final_eval.py --models unet,cnn_bilstm,unet_kan,convkan,deq
```

Training one fold model:

```bash
python src/train.py --model unet_kan --fold 0 --lr 3e-3 --seed 1 --tag lr3e-3
```

## Data

The ECG databases are public and are **not** redistributed here (except the small subsets needed by individual analyses, which are archived on Zenodo with their licences). `docs/DATA.md` lists versions, sources, licences and the expected folder layout under `data/raw/`.

## Large files

Model weights (321 checkpoints), record-level outputs and analysis inputs are archived on Zenodo together with a snapshot of this repository: https://doi.org/10.5281/zenodo.24554093

## Analysis plan

`docs/analysis_plan_v1.0_frozen_tr.md` is the dated analysis plan (version 1.0, 22 November 2026), byte-identical to the file whose SHA-256 digest is given in `docs/analysis_plan_v1.0_frozen_tr.sha256`. It is written in Turkish. Analyses that are not part of this plan are labelled as additional or post hoc in the paper and in the corresponding folders.

## Licence

- Code: MIT licence (`LICENSE`).
- Derived results in this repository (tables, counts, logs): CC BY 4.0.
- The ECG databases are not redistributed here; each remains under its publisher's licence (PhysioNet ODC-By and PhysioNet licences, CC0, CC BY 4.0, CC BY-NC 4.0 and CC BY-ND 4.0). Sources and versions are listed in `docs/DATA.md`.

## Citation

See `CITATION.cff`. Please cite the article once published.
