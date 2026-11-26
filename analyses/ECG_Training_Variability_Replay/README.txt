Training variability across eight ECG delineation architectures

The LUDB 1.0.1 native files included in native/ are from Kalyakulina A, Yusipov I,
Moskalenko V, Nikolskiy A, Kosonogov K, Zolotykh N and Ivanchenko M (2021).
Lobachevsky University Electrocardiography Database (version 1.0.1), PhysioNet.
Dataset: https://doi.org/10.13026/eegm-h675
Original scientific article: https://doi.org/10.1109/ACCESS.2020.3029211
Publisher: https://physionet.org/content/ludb/1.0.1/
The database is distributed under ODC-By 1.0. See native/LICENSE.txt.
The author's downloaded source ZIP had SHA-256
d03192c7361ab5deeaba3f0d46e274b9ecb0b74c9caf8964151e20f1b5a7df06.
Every bundled input can be checked against native/SHA256SUMS.txt.

Study layout and primary-benchmark sensitivity
For each training seed (1, 2, 3), each of eight architectures was trained on
five development folds. The 41 held-out participants were never in any training
or validation fold. All 12 leads from a participant were assigned together.
The five probability maps were averaged before decoding. All 24 ensembles
use the same 41 participants. There are 120 fold weights. The 60 archive
members are listed in inventory.txt. The 60 new fits have validation logs.
Results prefixed fold0 are a single-fold diagnostic, with the same test
participants as the principal five-member ensembles prefixed ensemble.
Protocol.txt was frozen before the first new fit. Its quoted "weighted
half-width" wording refers to the original crop placement; the Python
generator implements span-constrained crops and standard cross-entropy.

Read the manuscript-ready English methods/results in Manuscript_insert.md.
results/ensemble_scores.csv contains 3 seed values, mean, sample SD and ranks.
results/ensemble_contrasts.csv contains descriptive paired 95% intervals.
results/ensemble_subject_counts.csv contains per-participant TP, FN, FP by
model, seed, tolerance and landmark across the participant's 12 leads.
results/checkpoint_manifest.csv contains weight origins and SHA-256 values.
results/ensemble_training_variation.png and .pdf show the observed seed ranges.
The original primary H1/H2/H3 tests and independent head-only experiments
are separate analyses. Consider hardware and annotation-edge scorer version
when interpreting variability between the archived and added fits.

Homogeneous secondary panel
Fold 0 was also evaluated using only models fit on the same GPU with the
edge-tolerant validation score. Twelve additional fits replaced the archived
CPU-trained members of that fold. The other 12 were reused, giving 24
same-implementation single models across 8 architectures and 3 seeds.
The primary-benchmark panel plus this single-fold check comprises 132 distinct
weights, including 72 new fits.
Homogeneous_fold_insert.md, homogeneous_protocol.txt, homogeneous_weights/,
homogeneous_logs/ and results/homogeneous* document this comparison.
The underlying archive's timestamp metadata and correction text appear
in source/archived_timestamps.csv and source/scoring_amendment.txt.
Conditional bootstrap intervals resample test participants across fixed
seeds. They do not estimate a population distribution of training seeds.

Principal all-GPU five-fold panel (complete package)
The full panel uses the same implementation, hardware, crop schedules and
edge-tolerant validation criterion for all 8 architectures, 3 seeds and 5
folds. All 120 fold weights were trained on the GPU. The panel reuses the
60 new members of the primary-benchmark sensitivity analysis and replaces all
60 original CPU members. Including primary-benchmark CPU files there are 180
unique weights. The interim release only covers the earlier 132 files.
Harmonized_ensemble_insert.md supplies the manuscript text and two tables.
The main figure and all corresponding tabulations begin with
results/harmonized_ensemble_. See results/harmonized_ensemble_checkpoint_manifest.csv
for the resolved path, original weight, source, and paired schedule hashes
for each condition. results/harmonized_cpu_gpu_inference_audit.json separately
checks numerical differences when the fixed weights are evaluated on CPU.
File timestamps predating the primary-benchmark rule are circumstantial
evidence, not individual provenance records of the checkpoint selection code.

Replay
Use Python 3.11 or newer with numpy, scipy, torch, wfdb, matplotlib, psutil.
Installed versions used for the delivered analysis are in results/run_environment.json.
Use a CUDA-enabled PyTorch build for efficient raw inference or refitting.
The compact summaries can be reproduced without waveforms, GPU or torch:

    python analyse.py ensemble
    python figure.py ensemble
    python audit_results.py ensemble
    python analyse.py homogeneous
    python audit_results.py homogeneous
    python analyse.py harmonized_ensemble
    python audit_results.py harmonized_ensemble
    python figure.py harmonized_ensemble

Verify the source signals and independently re-score the model weights:

    python check_inputs.py
    python prepare.py --signals-only
    python train_panel.py --score-only --result-dir independent_inference

Compare the per-subject NPZ counts in independent_inference/ against the
distributed results/. CPU/GPU and library versions can affect borderline
samples. On a shared configuration the counts should reproduce; use the
published seed-1 table as an independent rounded aggregate anchor.
For a separate CPU run, python cpu_crosscheck.py replay uses one lead per
call for every architecture. python cpu_crosscheck.py compare summarizes
all 48 CPU–GPU score differences and all 11,808 subject/landmark cells.

To retrain the newly added models, use a NEW working copy, copy all .py,
protocol.txt, splits.json, chosen_lr.json and native/, and under weights/
copy just the 60 original CPU files named in inventory.txt. Then run:

    python prepare.py --native
    python train_panel.py --train
    python homogeneous.py
    python harmonized.py --folds 1 2 3 4
    python harmonized.py --score
    python analyse.py ensemble
    python analyse.py harmonized_ensemble

Optional implementation equivalence audit with the recovered archived trainer:

    python prepare.py --native
    python validate_native.py

The replay package has all three seeds. A completed fit is never overwritten.
The full training batch cache uses approximately 4 GB of additional disk
space and is regenerated from the raw data and seeds. Mismatched model keys,
subject counts, reference denominators, tolerance ordering and weight hashes
are checked in the analysis. Raw units are milliseconds and F1 percent;
absolute contrasts are percentage points.
