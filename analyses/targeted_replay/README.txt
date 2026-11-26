Targeted analyses of ECG delineation models

This archive contains the published derivation of paired training-run contrasts from record-level counts, internal channel-replacement experiments on LUDB lead II, and the second-processor latency and training-memory measurements. Channel weights are supplied once in the accompanying RDB_external_validation.zip. The three fold-zero DEQ and twin checkpoints are in profil/resources. Raw recordings are obtained from PhysioNet.

Environment

Python 3.12.10, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, wfdb 4.3.0, psutil 7.2.2 and PyTorch 2.14.1+cpu were used. Install the CPU build matching the platform using the PyTorch distribution instructions, then install the pinned analysis dependencies. The timing and peak-working-set programs use Windows APIs. Numerical and channel replays use PyTorch on CPU. In the source manuscript environment a locally installed wfdb distribution is sufficient.

python -m pip install -r requirements.txt
python replay.py verify

Data and trained weights

Place RDB_external_validation.zip beside the extracted replay directory or provide its path. From the extracted directory run:

python prepare_inputs.py --rdb-zip ../RDB_external_validation.zip

The command downloads the original PhysioNet LUDB v1.0.1 release, compares its checksum manifest to the pinned published version and materializes only 200 .hea files, 200 .dat files and 200 lead-II .ii annotation files. A previously downloaded official release ZIP can be passed with --ludb-zip. A .dat file contains the original 12-lead recording; the analyses use lead II. Existing files with different hashes are never overwritten. The same command checks the 30 RDB-package training weights against both manifests and extracts them to channel_weights/. This experiment uses no RDB raw signals. Each learned model was trained on LUDB.

Data attribution: Kalyakulina A, Yusipov I, Moskalenko V, Nikolskiy A, Kosonogov K, Zolotykh N, Ivanchenko M. Lobachevsky University Electrocardiography Database v1.0.1. PhysioNet, 2021. DOI 10.13026/eegm-h675.
https://physionet.org/content/ludb/1.0.1/
https://physionet.org/files/ludb/1.0.1/SHA256SUMS.txt

Replay

python replay.py paired
python replay.py channels
python replay.py hybrid
python replay.py latency
python replay.py memory

The first command needs only the bundled head_counts.csv. It uses the cleaned BUT P reference counts, paired record IDs and 10,000 subject/source bootstrap draws. Channels and hybrid each reselect channels on the archived validation folds, apply the interventions to all 41 held-out subjects, and check every recomputed cell of the three published CSV files. Channels compares the KAN-only and GELU heads; hybrid repeats the protocol for the full U-Net-KAN. CPU timings and memory allocations depend on the processor, software stack and other host activity; their reference results and methods are in profil/. replay.py writes to recomputed/<analysis>/, away from the reference CSVs. Use --out PATH to change that root.

SHA256SUMS.txt covers every original member except the checksum list itself. CPU checkpoints, source model definitions and subject split also have individual identifiers in profil/profile_summary.json. channel_weights_SHA256SUMS.txt lists the linked weights. Raw LUDB recordings and calibration arrays are excluded from the distribution.
