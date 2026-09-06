# Paper-reported results

This page collects the quantitative results reported in the RL-ScanIQA paper and its author-provided supplementary material. The repository does not include trained checkpoints, per-split logs, or raw benchmark data, so these tables are provided for reference and are **not presented as independently reproduced by this code checkout**.

## Evaluation protocol

- Benchmarks: CVIQD, OIQA, and JUFE.
- In-dataset evaluation: random 80/20 train/test split, repeated 10 times, reporting the mean.
- Cross-dataset evaluation: train on the full CVIQD or JUFE training source and test on the other datasets without target-domain fine-tuning.
- Metrics: Spearman rank-order correlation coefficient (SRCC) and Pearson linear correlation coefficient (PLCC); higher is better.
- Inference: average `K=15` scanpaths, each with `T=7` selected viewports.

## Main results

### In-dataset evaluation

| Dataset | SRCC | PLCC |
|:--|--:|--:|
| JUFE | 0.816 | 0.902 |
| OIQA | 0.941 | 0.967 |
| CVIQD | 0.970 | 0.970 |

### Cross-dataset evaluation

| Train | Test | SRCC | PLCC |
|:--|:--|--:|--:|
| CVIQD | OIQA | 0.901 | 0.913 |
| CVIQD | JUFE | 0.800 | 0.822 |
| JUFE | CVIQD | 0.771 | 0.802 |
| JUFE | OIQA | 0.755 | 0.833 |

## Ablation studies

### Scanpath source and joint training on JUFE

| Variant | SRCC | PLCC |
|:--|--:|--:|
| Human ground-truth scanpaths | 0.724 | 0.752 |
| Without joint training | 0.651 | 0.783 |
| Main model | 0.816 | 0.902 |

### Multi-level reward groups

| Variant | CVIQD SRCC / PLCC | OIQA SRCC / PLCC | JUFE SRCC / PLCC |
|:--|:--:|:--:|:--:|
| Without step-wise exploration reward (SER) | 0.952 / 0.960 | 0.903 / 0.942 | 0.754 / 0.766 |
| Without scanpath diversity reward (SDR) | 0.946 / 0.958 | 0.897 / 0.946 | 0.731 / 0.755 |
| Without task-aligned perceptual reward (TPR) | 0.921 / 0.933 | 0.874 / 0.916 | 0.720 / 0.771 |
| Main model in the ablation run | 0.968 / 0.977 | 0.941 / 0.967 | 0.816 / 0.902 |

> The CVIQD main-model value in this ablation table (`0.968 / 0.977`) differs from the paper's primary comparison table (`0.970 / 0.970`). Both are transcribed as reported and should be interpreted within their respective experiment tables.

### Fine-grained exploration reward components

| Variant | CVIQD SRCC / PLCC | OIQA SRCC / PLCC | JUFE SRCC / PLCC |
|:--|:--:|:--:|:--:|
| Without entropy term | 0.963 / 0.968 | 0.935 / 0.963 | 0.801 / 0.872 |
| Without `1 - SSIM` term | 0.958 / 0.963 | 0.931 / 0.956 | 0.791 / 0.852 |
| Without novelty term | 0.955 / 0.962 | 0.933 / 0.960 | 0.796 / 0.862 |
| Without equator-bias term | 0.967 / 0.973 | 0.938 / 0.965 | 0.808 / 0.886 |
| Main model in the ablation run | 0.968 / 0.977 | 0.941 / 0.967 | 0.816 / 0.902 |

### Distortion-space augmentation

| Variant | CVIQD → OIQA | CVIQD → JUFE | JUFE → CVIQD | JUFE → OIQA |
|:--|:--:|:--:|:--:|:--:|
| Without augmentation | 0.782 / 0.825 | 0.774 / 0.701 | 0.664 / 0.663 | 0.703 / 0.702 |
| Main model | 0.901 / 0.913 | 0.800 / 0.822 | 0.771 / 0.802 | 0.755 / 0.833 |

Each cell reports `SRCC / PLCC`.

### Augmentation-related losses

| Variant | CVIQD → OIQA | CVIQD → JUFE | JUFE → CVIQD | JUFE → OIQA |
|:--|:--:|:--:|:--:|:--:|
| Without consistency loss | 0.877 / 0.894 | 0.794 / 0.787 | 0.739 / 0.762 | 0.740 / 0.765 |
| Without triplet loss | 0.860 / 0.872 | 0.792 / 0.760 | 0.729 / 0.746 | 0.732 / 0.698 |
| Without cross-rank loss | 0.847 / 0.884 | 0.789 / 0.801 | 0.708 / 0.745 | 0.712 / 0.771 |
| Main model | 0.901 / 0.913 | 0.800 / 0.822 | 0.771 / 0.802 | 0.755 / 0.833 |

Each cell reports `SRCC / PLCC`.

## Distortion-space augmentation settings

At each strength level, one distortion type is sampled rather than stacking every distortion.

| Distortion | Weak | Mild | Strong |
|:--|:--|:--|:--|
| JPEG quality | `[85, 95]` | `[60, 75]` | `[20, 40]` |
| Motion-blur kernel | `[3, 7]` | `[7, 11]` | `[11, 19]` |
| Defocus radius | `[1, 2]` | `[2, 3]` | `[4, 6]` |
| Poisson rate | — | `[18, 30]` | `[6, 12]` |

Motion-blur angles are sampled randomly. The loss margins are `m1=0.02`, `m2=0.10`, and `m3=0.12`.

## Machine-readable summary

The main results are also provided in [results/paper_results.csv](../results/paper_results.csv).
