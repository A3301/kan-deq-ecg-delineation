Testing matched heads from two training sources on the same ISP lead-II records

The disk version of protocol.txt was saved and hashed on 25 November 2026 before applying any model in this additional analysis. The hash is embedded in run.py and results/inference_audit.json. The project had already used ISP for other evaluations. This extra experiment is retrospective.

The data are the unmodified ISP v2 release. Source files and model weights are acquired separately. The source metadata identify the CC BY-ND 4.0 license. Raw files and transformed signal arrays are not included here. The package distributes numerical evaluation results and code.

Avetisyan A, Khachaturov N, Asatryan A, Tigranyan S, Markin Y. ISP ECG delineation dataset, v2. Zenodo (2025). DOI 10.5281/zenodo.14679837. https://zenodo.org/records/14679837

Use Python 3.12 with numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, wfdb 4.3.0 and torch 2.14.1+cpu as in the accompanying targeted replay. No GPU is required. Extract this ZIP in a separate directory. Supply the paths to the two archived weight bundles in the wider computational package.

python replay_setup.py --rdb-validation ../RDB_external_validation.zip --reverse-training ../Reverse_source_training.zip
python run.py --isp inputs/isp_delineation_dataset.zip --ludb-weights inputs/LUDB --rdb-models inputs/RDB --out recomputed
python verify.py --isp inputs/isp_delineation_dataset.zip --results recomputed

If the public ZIP was downloaded already, add --isp PATH_TO_ISP_ZIP to replay_setup.py and use that same path for the other two commands. The setup checks the published ISP MD5 and original weight hashes. Every model file is loaded with weights_only=True and strict state-dictionary matching.

results/comparison.csv contains six boundary estimates plus their unweighted mean, for both training sources, each individual run and the average over runs, at both tolerances. F1 is in percent and differences are in percentage points. No clinical equivalence decision, between-source significance test or patient-level independence claim is implied. results/event_counts.csv is sufficient to reconstruct all 98 comparison rows and bootstrap limits. results/cohort.csv gives signal-file hashes and source annotation checks. verify.py independently rereads the 475 source headers, reference tuples, record counts and summary intervals. SHA256SUMS.txt identifies every file in this ZIP other than the list itself.
