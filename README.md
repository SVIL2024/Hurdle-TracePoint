# Hurdle-TracePoint

Official PyTorch implementation of **Hurdle-TracePoint: Latent Event Modeling with Structural-Zero Gating for Weakly Supervised Video Anomaly Detection**.

Hurdle-TracePoint learns temporal anomaly scores from video-level labels. It adds a latent event branch to [VadCLIP](https://github.com/nwpu-zxr/VadCLIP): the branch estimates whether an anomaly is present in a video, scores candidate events, and combines them into temporal class-specific coverage.

![Hurdle-TracePoint overview](assets/overview.png)

## How it works

The event branch has three main steps:

1. **Estimate presence.** Pool features from the complete video to estimate whether an anomaly, and each anomaly class, occurs.
2. **Score event candidates.** A gated recurrent unit (GRU) summarizes the visual prefix up to each possible start time. The branch scores candidates defined by their start, class, and duration.
3. **Build temporal scores.** Scale candidate weights by the presence estimates, then combine overlapping intervals into class-specific coverage and an overall event score.

The event branch shares CLIP features and class-text information with the VadCLIP frame detector. Each branch produces its own scores, and event losses update only the event branch. Presence estimation uses the complete video, so inference is offline.

The event model is in [src/tracepoint.py](src/tracepoint.py), and its connection to VadCLIP is in [src/model.py](src/model.py).

## Repository Layout

```text
Hurdle-TracePoint/
|-- assets/                 # README figures
|-- src/                    # Training, evaluation, model, CLIP, and utility code
|-- tests/                  # Unit tests for the TracePoint event process
|-- list/                   # Instructions for local dataset lists and annotations
|-- LICENSE                 # Apache License 2.0 for project files
|-- THIRD_PARTY_NOTICES.md  # Notices for vendored third-party components
|-- requirements.txt        # Python dependencies
`-- README.md
```

## Environment

Use Python 3.10 or newer. A CUDA-capable GPU is recommended for training.

```bash
git clone https://github.com/SVIL2024/Hurdle-TracePoint.git
cd Hurdle-TracePoint
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If your CUDA driver, GPU, or platform differs, install a PyTorch build matching your machine first, then install the remaining packages from `requirements.txt`. The CLIP implementation downloads its pretrained weights on first use.

Check PyTorch and CUDA visibility:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda build:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
PY
```

All commands below run from the repository root.

## Data Preparation

The project expects pre-extracted CLIP ViT-B/16 features and dataset annotations for:

  * UCF-Crime
  * XD-Violence

Dataset files and feature files are external to this repository. Download and use them only under their applicable terms.

| Resource | Download | Access code |
|---|---|---|
| Dataset package | [Quark Drive](https://pan.quark.cn/s/b57edbb83bd4) | `TwtL` |
| UCF-Crime and XD-Violence checkpoints | [Quark Drive](https://pan.quark.cn/s/72cde99ab1a9) | `xbCF` |

### Feature lists

Each `.npy` feature file contains a `float32` array of shape `(T, 512)`, where `T` is the number of snippets. Evaluation assumes 16 video frames per snippet.

Create training and test CSV files with two columns, `path` and `label`. Paths can be absolute or relative to the repository root. These CSV files are ignored by Git and must not be committed.

UCF-Crime example:

```csv
path,label
features/ucf/normal_001.npy,Normal
features/ucf/abuse_001.npy,Abuse
```

UCF-Crime labels use the names in the [training script's label map](src/ucf_train.py), including the capitalized normal label `Normal`.

XD-Violence example:

```csv
path,label
features/xd/normal_001.npy,A
features/xd/fighting_001.npy,B1
features/xd/multiple_001.npy,B1-B2
```

XD-Violence uses `A` for normal videos, `B1` for fighting, `B2` for shooting, `B4` for riot, `B5` for abuse, `B6` for car accident, and `G` for explosion. Join multiple anomaly labels with `-`, as in `B1-B2`.

### File locations

By default, the following files are read from `list/`:

| Input | UCF-Crime | XD-Violence |
|---|---|---|
| Training CSV | `ucf_CLIP_rgb.csv` | `xd_CLIP_rgb.csv` |
| Test CSV | `ucf_CLIP_rgbtest.csv` | `xd_CLIP_rgbtest.csv` |
| Frame labels | `gt_ucf.npy` | `gt.npy` |
| Temporal intervals | `gt_segment_ucf.npy` | `gt_segment.npy` |
| Interval class labels | `gt_label_ucf.npy` | `gt_label.npy` |

Keep test videos in the same order as the annotation arrays. Frame labels follow the concatenated test-video order; interval and class-label arrays have corresponding entries for each video.

For other locations, use `--train-list`, `--test-list`, `--gt-path`, `--gt-segment-path`, and `--gt-label-path`.

## Pre-trained Models

Place a downloaded checkpoint outside the repository and pass its path through `--model-path` or `--warm-start-path`. Checkpoints are not mirrored in Git.

## Training

The following commands use the main event configuration: both presence gates, seven duration choices, and a visual prefix without feedback from earlier event predictions.

UCF-Crime:

```bash
python src/ucf_train.py \
  --tracepoint --tracepoint-hurdle-gate --tracepoint-no-history \
  --output-dir outputs/ucf_hurdle
```

XD-Violence:

```bash
python src/xd_train.py \
  --tracepoint --tracepoint-hurdle-gate --tracepoint-no-history \
  --output-dir outputs/xd_hurdle
```

| Option | Purpose |
|---|---|
| `--tracepoint` | Enable the event branch alongside VadCLIP. |
| `--tracepoint-hurdle-gate` | Enable global and class-specific presence gates. |
| `--tracepoint-no-history` | Use the visual prefix without recurrent feedback from previous event candidates. |

The default duration choices are `1, 2, 4, 8, 16, 32, 64`. Each run trains for 10 epochs by default.

Each output directory may contain:

```text
outputs/ucf_hurdle/
├── best.pth          # Selected model checkpoint
├── last.pth          # Latest model state
├── run_config.json   # Training arguments
└── metrics.jsonl     # Evaluation records and training summaries
```

Use a separate output directory for each run. To run comparisons:

* **Raw TracePoint:** omit `--tracepoint-hurdle-gate` from the training and evaluation commands.
* **VadCLIP alone:** omit the event-branch flags.

## Evaluation

These examples evaluate the checkpoints produced by the training commands above:

```bash
python src/ucf_test.py \
  --model-path outputs/ucf_hurdle/best.pth \
  --tracepoint --tracepoint-hurdle-gate --tracepoint-no-history

python src/xd_test.py \
  --model-path outputs/xd_hurdle/best.pth \
  --tracepoint --tracepoint-hurdle-gate --tracepoint-no-history
```

For a downloaded checkpoint or a different model variant, use the gate, history, duration, and architecture settings from its training configuration.

The printed frame-level metrics refer to the two VadCLIP frame-detector scores:

| Printed metric | Score being evaluated |
|---|---|
| `AUC1`, `AP1` | Visual classification score |
| `AUC2`, `AP2` | Visual-text alignment score |

The released evaluator calls localization with `excludeNormal=False`. Therefore, its default localization mAP follows the all-class protocol: UCF-Crime includes `Normal`, and XD-Violence includes `A`. Anomaly-only numbers must be reported separately with the corresponding evaluation protocol.

## Main Results

The following values are reported for the paper's anomaly-only localization protocol using the seven-duration configuration. They are not the default all-class localization values printed by `src/ucf_test.py` and `src/xd_test.py`; use the same checkpoint, configuration, and anomaly-only evaluator before comparing results.

| Metric | UCF-Crime | XD-Violence |
|---|---:|---:|
| Coverage mAP, ungated → gated (%) | 1.94 → 4.04 | 11.74 → 14.12 |
| Normal-frame false-positive rate (FPR) at 80% recall, ungated → gated (%) | 57.12 → 8.65 | 26.68 → 6.00 |

The frame-detector scores and event-branch scores are reported separately.

For all available options:

```bash
python src/ucf_train.py --help
python src/ucf_test.py --help
python src/xd_train.py --help
python src/xd_test.py --help
```

## Tests

Run the unit tests without downloading the datasets:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The tests cover event weights, temporal coverage, padding, recurrent state across chunks, gradients, and checkpoint helpers.

## Privacy and Release Policy

Do not commit absolute local paths, personal contact information, shell transcripts, checkpoints, generated artifacts, or dataset files. Review `git status` and the staged file list before every public push.

## Acknowledgements and License

This implementation builds on [VadCLIP](https://github.com/nwpu-zxr/VadCLIP) and [OpenAI CLIP](https://github.com/openai/CLIP). We also acknowledge [XDVioDet](https://github.com/Roc-Ng/XDVioDet) and [DeepMIL](https://github.com/Roc-Ng/DeepMIL).

Project code is distributed under the [Apache License 2.0](LICENSE). The vendored CLIP components retain their original MIT license; see `THIRD_PARTY_NOTICES.md`.
