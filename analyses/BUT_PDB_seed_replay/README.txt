Pathological P-peak detection on the BUT PDB across three fitted seeds

The source-selected, five-development-fold CNN-BiLSTM ensemble is compared with full U-Net-KAN, ConvKAN and implicit DEQ. All four architectures were trained on LUDB. All three trained seeds are evaluated on the same external data. A seed contains five networks whose predictive probabilities are averaged. The 12 ensembles comprise 60 fold weights. The initial cleaned H3 decision remains the frozen hypothesis result. This extension examines reproducibility under training seeds and matched subject clusters. Training runtime and validation-checkpoint conventions of legacy and new fits are documented in the accompanying training replay.

Data and attribution
Maršánová L, Nemcova A, Smisek R, Smital L, Vitek M (2021). Brno University of Technology ECG Signal Database with Annotations of P Wave, version 1.0.0. PhysioNet, DOI 10.13026/hwvj-5b53, https://physionet.org/content/but-pdb/1.0.0/ . We distribute the required unmodified .hea, .dat, .pwave, README.txt, LICENSE.txt and SHA256SUMS.txt files under the release CC BY 4.0 license (source/native/LICENSE.txt). We cite the source records of the MIT-BIH Arrhythmia Database, MIT-BIH Supraventricular Arrhythmia Database and Long Term AF Database as listed in the dataset README.

Models were applied to both traces in each of 50 two-minute recordings. The 50 segments trace to 45 underlying source recording IDs, with 5,429 annotated P peaks shared by the two leads. The segment numbers 08 and 48 have a no-event sentinel in their native .pwave files, so they contribute no reference peaks and every prediction there is counted as false positive. P annotations are duplicated across leads for scoring rather than treated as independent observations. Bootstrap sampling uses the 45 underlying source identifiers. The reference tolerance was originally converted to the nearest 250-Hz tick. The complementary native_ms series retains exact native reference coordinates. Source metadata document an overlap between segments 05 and 26 from MIT-BIH Arrhythmia Database record 118. The exploratory nonoverlap49 sensitivity keeps the earlier numbered segment. Results conditional on the three fixed sets of networks do not quantify uncertainty over arbitrary future network initialization or new health-care sites.

Statistical replay with the supplied outputs
Python 3.12, NumPy 2.5.3 and pandas 3.0.6 were used. Extract this ZIP into an empty directory, then use:
python summarize.py
python verify.py

Waveform and model replay, optional
Install CPU PyTorch (tested 2.14.1+cpu), SciPy 1.18.1, WFDB 4.3.0 and the numeric packages above. In a fresh extraction run:
python get_weights.py PATH/ECG_Training_Variability_Replay.zip
python run.py weights --threads 2
python summarize.py
python verify.py

run.py skips an ensemble with a matching recorded weight, input and output digest. To force new predictions use a separate copy with its parts/ folder moved out of the analysis directory first. Each ensemble is evaluated on each lead separately with the original batching implementation. DEQ can depend on the membership of a batch of windows within a lead. If the training ZIP is already unpacked, the weights directory can be supplied directly.

Interpretation of files
results/contrasts.csv reports the individual seeds and the equal average of their pooled-record F1 values. Confidence limits are from 10,000 paired source-record bootstrap resamples, with Holm-adjusted empirical two-sided tail areas across each fixed family of three challengers. The "mean" rows are not probability averages across seeds. cohort all50 is the principal analysis, nonoverlap49 an exploratory sensitivity. historical_grid reproduces the parent H3 convention. native_ms applies exact physical tolerances using source sampling times. results/events.csv contains all segment/lead error counts, results/predictions.csv each predicted P-peak time in 4-ms ticks, and source/references.csv the single list of reference peaks in native samples for each original segment. results/negative_reference_events.csv identifies the contributions from the two no-P records. results/weights.json lists the 60 input weight hashes. protocol.txt and overlap_addendum.txt record the analysis choices.
