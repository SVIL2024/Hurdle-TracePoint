# Hurdle-TracePoint

Official PyTorch implementation of **Hurdle-TracePoint: Presence-Gated Event Scoring for Weakly Supervised Video Anomaly Detection**.

Hurdle-TracePoint adds an event branch to [VadCLIP](https://github.com/nwpu-zxr/VadCLIP). It first estimates whether an anomaly is present in the complete video, then uses that estimate to gate event candidates defined by their start time, class, and duration. Overlapping candidates are combined into class-specific temporal coverage and an overall event score.

![Hurdle-TracePoint overview](assets/overview.png)

## How it works

1. **Estimate presence.** Attention pooling produces global and class-specific presence gates from the complete video.
2. **Score event candidates.** A GRU encodes the visual prefix up to each possible start time and scores candidate events across classes and durations.
3. **Build temporal scores.** The presence gates weight the candidates, which are then combined into class coverage and an overall event score.

The event branch shares CLIP features and class-text information with the companion VadCLIP detector. It is enabled explicitly from the command line, while the original VadCLIP scoring path remains available separately.

## Installation

Use Python 3.10 or newer. A CUDA-capable GPU is recommended for training.

```bash
git clone https://github.com/SVIL2024/Hurdle-TracePoint.git
cd Hurdle-TracePoint
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your hardware if the package in your environment does not provide one. All commands below run from the repository root.

## Data and checkpoints

The repository uses external UCF-Crime and XD-Violence data, pre-extracted CLIP features, and model checkpoints. Download them under their applicable terms.

| Resource | Download | Access code |
|---|---|---|
| Dataset package | [Quark Drive](https://pan.quark.cn/s/b57edbb83bd4) | `TwtL` |
| UCF-Crime and XD-Violence checkpoints | [Quark Drive](https://pan.quark.cn/s/72cde99ab1a9) | `xbCF` |

Do not commit videos, annotations, feature files, checkpoints, or experiment outputs.

### Feature lists

The scripts read pre-extracted CLIP ViT-B/16 features. Each `.npy` file contains a `float32` array of shape `(T, 512)`, with 16 video frames represented by each snippet. Create local CSV files with two columns:

```csv
path,label
features/ucf/normal_001.npy,Normal
```

By default, the following files are read from `list/`:

| Input | UCF-Crime | XD-Violence |
|---|---|---|
| Training CSV | `ucf_CLIP_rgb.csv` | `xd_CLIP_rgb.csv` |
| Test CSV | `ucf_CLIP_rgbtest.csv` | `xd_CLIP_rgbtest.csv` |
| Frame labels | `gt_ucf.npy` | `gt.npy` |
| Temporal intervals | `gt_segment_ucf.npy` | `gt_segment.npy` |
| Interval class labels | `gt_label_ucf.npy` | `gt_label.npy` |

Keep test videos in the same order as the annotation arrays. For other locations, use `--train-list`, `--test-list`, `--gt-path`, `--gt-segment-path`, and `--gt-label-path`.

## Training

Run the main Hurdle-TracePoint configuration:

```bash
python src/ucf_train.py --tracepoint --tracepoint-hurdle-gate --tracepoint-no-history --seed 234 --output-dir outputs/ucf_hurdle
python src/xd_train.py  --tracepoint --tracepoint-hurdle-gate --tracepoint-no-history --seed 234 --output-dir outputs/xd_hurdle
```

The output directory contains the selected checkpoint (`best.pth`), run settings (`run_config.json`), and metrics (`metrics.jsonl`). Omit the three `--tracepoint*` options to run the companion VadCLIP baseline.

## Evaluation

```bash
python src/ucf_test.py --model-path outputs/ucf_hurdle/best.pth --tracepoint --tracepoint-hurdle-gate --tracepoint-no-history
python src/xd_test.py  --model-path outputs/xd_hurdle/best.pth  --tracepoint --tracepoint-hurdle-gate --tracepoint-no-history
```

When evaluating a downloaded checkpoint, use the same event-branch options and duration settings used during training.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Acknowledgements

This project builds on [VadCLIP](https://github.com/nwpu-zxr/VadCLIP) and cites XDVioDet and DeepMIL as methodological references. The vendored CLIP components are based on the official OpenAI CLIP implementation; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licensing information.

## License

Project files are distributed under the Apache License 2.0 in [LICENSE](LICENSE), except where a component notice states otherwise.
