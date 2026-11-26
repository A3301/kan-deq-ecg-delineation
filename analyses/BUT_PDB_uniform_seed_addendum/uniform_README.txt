Homogeneous GPU seed sensitivity for the BUT PDB extension

Extract BUT_PDB_seed_replay.zip into a fresh folder, then extract BUT_PDB_uniform_seed_addendum.zip into the same folder. The second ZIP adds the homogeneous outputs under uniform/. It requires the first package's dataset, shared model source and scoring code. No prior results are replaced. The original 50-segment/45-source pathological P-peak analysis retains its primary-benchmark decision.

Run the statistical replay of the uniform ensemble set with:
python uniform.py

To rerun waveform inference first extract ECG_Training_Variability_Replay.zip separately. Its top-level inventory.txt and the weights/ and homogeneous_weights/ subfolders identify the frozen training fits. Move uniform/parts/ to a backup folder in your replay copy to force fresh inference of the changed panels, then run:
python uniform.py --training PATH_TO_EXTRACTED_TRAINING_ARCHIVE --threads 2

The uniform evaluation retains the original three seeds and five folds. The 30 primary-benchmark CPU fold weights (four seed-1 ensembles plus full U-Net-KAN and implicit DEQ seed 2) were replaced by new weights trained using the shared GPU and edge-tolerant validation implementation. The other 30 fold weights were already trained with that same implementation. If every SHA-256 in a five-fold ensemble matches the mixed-history set, its output is reused after verification of prediction-file digests. All changed ensembles are evaluated from signals. The analysis is conditional on these fitted seeds and reuses the 45 source records already examined.

uniform/results/contrasts.csv includes seed-wise and average contrasts, exact native timing, primary-benchmark grid timing and the overlap sensitivity. uniform/results/panel_lineage.csv gives the mixed and homogeneous model weight hashes at every fold and indicates reuse. uniform/results/weights.json records the full inference manifest for each ensemble. uniform/results/predictions.csv and events.csv allow statistical and independent score recomputation. Source terms and attribution are in the parent ZIP's README.txt.
