Berlin PTB external validation of the six frozen RDB source models

Scope and design
This package evaluates the QT duration of one manually selected beat per recording, using lead II, with true QRS onset available for beat identification only. protocol.txt was fixed locally on 25 November 2026 before applying any model to PTB. SHA-256 f000640816ba7b07191bbf2d25e8e1bfa54ceb2b74b5ffffbaa9574ad7b9cdfc. Models and checkpoints had already been selected on the RDB source data. The Latin-letter abbreviation PTB refers to the original Berlin PTB Diagnostic ECG Database. PTB-XL and the QT Database are distinct datasets.

Inputs, credit and permissions
The 549 ten-second waveform clips bundled here are the first 10,000 native samples of lead II from PhysioNet PTB Diagnostic ECG Database version 1.0.0. These digital arrays keep the original integer samples. source/cohort.csv stores the gain, baseline, recording identifier, patient directory and checksums needed to recover physical units. Credit to the PTB database contributors and Michael Oeff, University Clinic Benjamin Franklin, Berlin. Original data DOI 10.13026/C28C71. Source https://physionet.org/content/ptbdb/1.0.0/ . The database is available under the Open Data Commons Attribution License v1.0, https://opendatacommons.org/licenses/by/1-0/ . The full source record files have not been bundled.

The manual reference is Table 4, Additional file 1, from Christov I, Dotsinsky I, Simova I, Prokopova R, Trendafilova E, Naydenov S. Dataset of manually measured QT intervals in the electrocardiogram. BioMed Eng OnLine. 2006;5:31. https://doi.org/10.1186/1475-925X-5-31 . It is distributed by the authors under Creative Commons Attribution 2.0, https://creativecommons.org/licenses/by/2.0/ . source/reference.doc is the original publisher file and source/reference.docx is its Office Open XML conversion. source/manual_reference.csv is the tabular transcription. Each row is a recording. There are five Q onset values, the published Q consensus, five T end values and the published T consensus, all in native milliseconds. Patient prefixes from the merged table cells are carried forward. Original attachment https://static-content.springer.com/esm/art%3A10.1186%2F1475-925X-5-31/MediaObjects/12938_2006_174_MOESM1_ESM.doc .

The five readers were four cardiologists and a biomedical engineer. The source describes elective re-reading when observers disagreed and occasional multi-lead information for T end. Published columns labelled Med exactly equal the rounded mean of the five reported inputs at both landmarks in every usable record. In the secondary, post hoc annotation audit, we repeat the primary endpoint using literal component-wise medians. The five final marks are adjudicated and correlated. The recording patient285/s0544_re has no interpretable ECG or positive reference, giving 548 eligible records from 289 participant directories. Models predicted every released clip before the reference was used to evaluate the selected beat.

Replay, offline
Use Python 3.12, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1 and torch 2.14.1+cpu for the tested environment. Models use CPU. requests is used only for the public source download; wfdb 4.3.0 is optional for independent verification of raw decoding and for the aligned waveform screen. Unzip in an empty directory. Run each command there. Predictions and scores will be rewritten into results/.

python extract_reference.py
python evaluate.py
python summarize.py
python audit_reference.py
python verify.py

Re-acquire the original signals and verify every checksum, optional
python download.py
python prepare_clips.py
python audit_reference.py --wfdb

This downloads the native 12-standard-lead .dat file and its .hea file for each record. The released digital container includes 12 interleaved channels. The separate Frank .xyz file is not needed for lead II. The optional WFDB audit rereads all 549 records and independently verifies all 5,490,000 native lead-II sample values. To regenerate the converted publisher table using Microsoft Word on Windows, run convert_reference.ps1, then python extract_reference.py. The ZIP has everything needed for the main analysis offline.

Check for highly similar aligned signals in the project's other cohorts, optional
python source_overlap.py PATH_TO_RDB_TRAINING/rdb_x.npy PATH_TO_ISP/isp_delineation_dataset.zip PATH_TO_LUDB/native_data

This check uses each of 2,398 RDB inputs, 475 ISP records and 200 LUDB records and the 549 original PTB clips. It tests central nine-second aligned absolute Pearson correlation after resampling to 50 Hz. The available source archives have no common patient identifiers. This screen does not establish whether a participant underwent a different examination or whether a trace with a different start time had been released elsewhere.

Files
results/summary.csv reports 10,000 patient-cluster percentile intervals for all fixed-seed averages. Percent values are in percent, head contrasts in percentage points. Other outputs are milliseconds. Each contrast is KAN minus GELU. results/by_seed.csv gives the component fits. results/counts.csv includes integer counts and missingness. results/per_record.csv and results/predicted_runs_at_250Hz.csv allow independent re-scoring without the model software. results/common_set.csv identifies beats returned by every fit. results/readers.csv gives each case's final reader QT range. results/posthoc_median_sensitivity.json gives the audit using freshly calculated medians. results/source_overlap.json gives aligned-duplicate screen results. results/denominators.json provides the record and participant counts. results/inference_manifest.json identifies the computational inputs. SHA256SUMS.txt covers all supplied files except itself. The thresholds, pairing rules and exclusions are documented in protocol.txt.
