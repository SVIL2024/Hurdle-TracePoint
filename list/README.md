# Local dataset files

Dataset-specific files are deliberately not included in this public code release. Obtain the datasets and extracted CLIP features under their applicable terms, then create local CSV files with two columns:

```csv
path,label
path/to/feature.npy,Normal
```

The `path` values may be absolute on the local machine, but those CSV files are ignored by Git and must never be committed. Supply them with `--train-list` and `--test-list`.

Evaluation additionally needs the corresponding ground-truth NumPy files. Supply them with `--gt-path`, `--gt-segment-path`, and `--gt-label-path`.

The code does not download or redistribute UCF-Crime, XD-Violence, extracted features, or model checkpoints.
