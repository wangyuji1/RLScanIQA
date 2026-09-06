# Datasets

RL-ScanIQA is evaluated on three panoramic image-quality benchmarks. Raw data are not included in this repository because dataset licenses and distribution terms belong to their original providers.

| Dataset | Images | References | Distortion profile |
|:--|--:|--:|:--|
| CVIQD | 528 | 16 | JPEG, AVC, and HEVC compression |
| OIQA | 320 | 16 | JPEG, JPEG2000, Gaussian blur, and Gaussian noise |
| JUFE | 1,032 | 258 | Non-uniform regional distortions and eye-movement data |

## Recommended local layout

```text
datasets/
├── CVIQD/
│   ├── images/
│   └── metadata/
├── OIQA/
│   ├── images/
│   └── metadata/
└── JUFE/
    ├── images/
    └── metadata/
```

Keep this directory outside Git or under the ignored `datasets/` path.

## Preparing pair supervision

The reference trainer expects `img1,img2,Q1,Q2`. A pair builder should:

1. Read the official MOS annotations.
2. Keep both images within the current training split.
3. Sample pairs with a useful range of MOS differences.
4. Store paths relative to `--img_root`.
5. Never mix test images into training pairs.

Example:

```csv
img1,img2,Q1,Q2
images/example_a.jpg,images/example_b.jpg,64.67,16.15
```

The synthetic path (`--pairs_csv ''`) exists only for software smoke testing.
