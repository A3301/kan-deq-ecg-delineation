# Data sources

All databases are public. They are not redistributed in this repository. Download them from the sources below and place them under `data/raw/` as listed in `src/paths.py`. Individual analyses in `analyses/` contain their own acquisition scripts and checksum lists.

| Database | Version | Use in the study | Source |
|---|---|---|---|
| LUDB (Lobachevsky University Electrocardiography Database) | 1.0.1 | Development (training, cross-validation, internal test); reverse-direction test | https://physionet.org/content/ludb/1.0.1/ (doi:10.13026/eegm-h675) |
| QT Database (QTDB) | 1.0.0 | External test (annotator 1); reader agreement | https://physionet.org/content/qtdb/1.0.0/ |
| BUT PDB | 1.0.0 | Primary external endpoint (P peaks) | https://physionet.org/content/but-pdb/1.0.0/ |
| MIT-BIH Arrhythmia Database P-wave annotations | 1.0.0 | External test (P peaks) | doi:10.13026/C2108F |
| Zhejiang (idiopathic ventricular arrhythmia, 334 patients) | — | External test (global labels) | Signals: Zheng et al., Sci. Data 7 (2020) 98, doi:10.1038/s41597-020-0440-8; labels: doi:10.6084/m9.figshare.14035196.v1 |
| ISP ECG delineation dataset | — | External test (global labels); common test of both training sources | doi:10.5281/zenodo.14679837 |
| Resting ECG Segmentation Dataset (RDB) | v4 | Transport test; training source of the reverse-direction experiment | doi:10.6084/m9.figshare.28892186.v4 |
| PTB Diagnostic ECG Database | 1.0.0 | Berlin QT reference recordings | https://physionet.org/content/ptbdb/1.0.0/ |
| Christov et al. QT reference coordinates | — | Five-reader QT reference | Additional file 1 of doi:10.1186/1475-925X-5-31 |

All signals are resampled to 250 Hz and filtered with a 0.5–40 Hz zero-phase band-pass filter before use (see `src/data.py`).
