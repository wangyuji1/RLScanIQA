# Reproducibility guide

## Release scope

This repository packages the reusable RL-ScanIQA components and a readable, runnable reference trainer. It intentionally does not redistribute benchmark datasets, private review documents, or pretrained checkpoints.

The paper's exact numbers additionally depend on author-side preprocessing, ten random splits, the frozen DINOv2 feature pipeline, training infrastructure, and model checkpoints. Consequently, the included synthetic smoke test validates code execution only; it is not a benchmark reproduction.

## Environment

Minimum supported environment:

- Python 3.9+
- PyTorch 1.12+ (PyTorch 2.x recommended)
- torchvision 0.13+
- NumPy
- Pillow
- OpenCV Python

Install with:

```bash
python -m pip install -e .
```

## Paper configuration

| Component | Setting |
|:--|:--|
| Candidate viewports | `8 × 4 = 32` |
| Viewport field of view | `90° × 90°` |
| Viewport resolution | `224 × 224` |
| Policy | GRU-based autoregressive policy |
| Policy optimizer | Adam, learning rate `3e-4` |
| Quality-assessor optimizer | Adam, learning rate `1e-4` |
| Adam betas | `(0.9, 0.999)` |
| PPO clip schedule | `0.20 → 0.10` |
| Entropy schedule | `0.02 → 0.005` |
| Discount / GAE | `γ=0.99`, `λ=0.95` |
| Gradient clipping | maximum L2 norm `1.0` |
| Training duration | 300 epochs |
| Batch size | 4 |
| Inference | `K=15`, `T=7`, average scanpath scores |

## Dataset protocol

The reported experiments use CVIQD, OIQA, and JUFE. Obtain each dataset from its original provider and follow its license and citation requirements.

For in-dataset evaluation:

1. Randomly split each dataset into 80% training and 20% testing.
2. Repeat the split and training procedure ten times with fixed, recorded seeds.
3. Report mean SRCC and PLCC over the ten test splits.

For cross-dataset evaluation:

1. Train on the full source dataset (CVIQD or JUFE).
2. Evaluate directly on each target dataset without target-domain fine-tuning.
3. Keep model and inference hyperparameters fixed.

JUFE provides four MOS values per image from two starting points and two viewing durations. The paper averages all four values into one MOS label for the single-label blind-IQA setting.

## Pair CSV used by the reference trainer

The included trainer consumes pairs so that regression and ranking losses can be computed in the same batch:

```csv
img1,img2,Q1,Q2
images/example_a.jpg,images/example_b.jpg,64.67,16.15
```

Both image paths are resolved relative to `--img_root`. Images should use equirectangular projection with an approximate `1:2` height-to-width ratio.

## Suggested experiment checklist

- Record Python, PyTorch, CUDA, GPU, and driver versions.
- Store all ten split files and random seeds.
- Verify that MOS orientation is consistent (higher means better quality).
- Keep DINOv2 frozen and document the exact model/weight revision.
- Normalize DINOv2 inputs consistently during training and evaluation.
- Use `K=15`, `T=7` for reported inference unless running a declared ablation.
- Save configuration, checkpoints, and per-split SRCC/PLCC logs.
- Report both the mean and per-split metrics.

## Module map

| Module | Purpose |
|:--|:--|
| `viewport_discretization.py` | Candidate centers and ERP-to-viewport projection |
| `policy.py` | GRU scanpath policy and value head |
| `advantages.py` | Generalized advantage estimation |
| `ppo_buffer.py` | Time-major rollout storage |
| `ppo.py` | PPO utilities and schedules |
| `rewards.py` | Exploration, diversity, MSE, and rank rewards |
| `data_augmentation.py` | Weak/mild/strong distortion pipelines |
| `losses.py` | Regression, ranking, consistency, triplet, and cross-rank losses |

## Current limitations

- Pretrained checkpoints and exact author-side training logs are not included.
- The standalone trainer is a compact reference path and does not replace the complete internal experiment launcher used for all paper tables.
- `torch.hub` is used for the optional DINOv2 path and may require network access on first use.
- Benchmark evaluation and split orchestration should be added when the corresponding licensed datasets are available locally.
