# DAPO and three-experiment comparison

Scan date: 2026-11-05. All scores below are percentages; differences are percentage points.

## Main findings

- Spectral improves final-epoch image metrics over the composed baseline, but decreases pixel AP, pixel F1-max and AUPRO.
- All five region reports are byte-for-byte identical to the matching baseline reports (verified with SHA-256). They do not provide independent evidence of the region objective’s effect. This could reflect export, path or checkpoint selection errors; the cause cannot be determined locally.
- The DAPO reproduction values in `reproduce.txt` use a different evaluation protocol from the project’s pixel results. Multi-type AP/F1 must not be compared directly with binary AP/F1-max.

## Evidence and experiment setup

Read all 15 JSON reports from `baseline-evaluation/`, `spectral-evaluation/` and `region-evaluation/`. Each uses MPDD, the same six product categories and the same ordered 20-name vocabulary. Recomputed every macro metric from the category values; all matched. Epoch identity comes from filenames only.

| Experiment | Config | Spectral fusion | Region loss |
|---|---|---|---|
| Composed baseline | `configs/baseline.json` | Disabled | Disabled |
| Spectral | `configs/spectral.json` | Enabled | Disabled |
| Spectral + region | `configs/spectral_region.json` | Enabled | Weight 0.1, temperature 0.07 |

All configs specify MVTec source training, five epochs, batch size 8, learning rate 0.001, seed 42, and loss weights 4 for segmentation and 1 for classification. These are configured settings: no local checkpoints, saved run `experiment.json`, training `metrics.jsonl`, or SLURM logs were found to verify actual run settings.

## Final epoch: epoch_4

Epoch_4 is the fifth epoch. Comparing a common final epoch avoids selecting checkpoints from MPDD test performance.

| Metric | Baseline | Spectral | Region (duplicate baseline) | Spectral − baseline |
|---|---:|---:|---:|---:|
| Image AUROC | 76.13 | 77.23 | 76.13 | +1.10 |
| Image AP | 79.43 | 82.78 | 79.43 | +3.35 |
| Image F1-max | 80.89 | 84.17 | 80.89 | +3.28 |
| Pixel AUROC | 93.52 | 93.76 | 93.52 | +0.24 |
| Pixel AP | 26.00 | 24.38 | 26.00 | -1.61 |
| Pixel F1-max | 29.82 | 26.47 | 29.82 | -3.35 |
| Pixel AUPRO | 71.80 | 68.55 | 71.80 | -3.24 |

Spectral’s largest final image AUROC gains are metal_plate (+18.09 pp) and connector (+9.76 pp). It loses on bracket_black (−8.18 pp), bracket_brown (−9.73 pp), bracket_white (−2.00 pp), and tubes (−1.35 pp). The macro improvement therefore does not imply consistent product-level gains.

## Epoch trajectories

| Experiment | Epoch | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel F1-max | Pixel AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 80.47 | 81.42 | 84.23 | 92.81 | 27.66 | 33.03 | 64.33 |
| baseline | 1 | 76.22 | 80.64 | 80.70 | 94.48 | 26.01 | 30.14 | 68.18 |
| baseline | 2 | 74.24 | 80.55 | 81.28 | 94.04 | 27.07 | 30.32 | 66.40 |
| baseline | 3 | 76.49 | 81.98 | 83.69 | 94.53 | 29.02 | 32.38 | 71.77 |
| baseline | 4 | 76.13 | 79.43 | 80.89 | 93.52 | 26.00 | 29.82 | 71.80 |
| spectral | 0 | 80.37 | 85.79 | 86.83 | 95.12 | 29.02 | 33.38 | 77.65 |
| spectral | 1 | 72.97 | 76.41 | 79.26 | 94.26 | 27.72 | 30.97 | 66.65 |
| spectral | 2 | 74.81 | 81.14 | 79.61 | 94.63 | 28.93 | 31.97 | 73.11 |
| spectral | 3 | 74.48 | 80.83 | 79.50 | 95.04 | 26.29 | 29.57 | 65.76 |
| spectral | 4 | 77.23 | 82.78 | 84.17 | 93.76 | 24.38 | 26.47 | 68.55 |

Region repeats the entire baseline trajectory exactly. Spectral epoch_0 has its highest observed value for all seven metrics. From epoch_0 to epoch_4 its image AUROC drops 3.14 pp, pixel AP drops 4.64 pp, and AUPRO drops 9.10 pp. This suggests worsening target performance over training, but training logs and repeated seeds are needed to investigate why. Epoch_0 is after the first training epoch, not an untrained checkpoint. Do not select it as a final checkpoint based on these target test results.

## DAPO reference values

The following values are transcribed from the local `reproduce.txt`, which describes a provided image. The paper and original raw reproduction reports were not independently verified. Only MPDD overlaps with the three experiment report sets.

| Binary metric | DAPO paper (local transcription) | DAPO reproduced EP4 | Baseline EP4 | Spectral EP4 | Region EP4 (duplicate) |
|---|---:|---:|---:|---:|---:|
| Image AUROC | 81.2 | 74.3 | 76.13 | 77.23 | 76.13 |
| Image AP | 83.6 | 79.8 | 79.43 | 82.78 | 79.43 |
| Pixel AUROC | 95.1 | 94.8 | 93.52 | 93.76 | 93.52 |
| Pixel AUPRO | 84.4 | 80.7 | 71.80 | 68.55 | 71.80 |

Numerically, spectral EP4 differs from reproduced DAPO EP4 by +2.93 pp image AUROC, +2.98 pp image AP, −1.04 pp pixel AUROC and −12.15 pp AUPRO. These are descriptive differences, not controlled estimates of architectural effects.

The local DAPO multi-type MPDD reference is paper AUROC/AP/F1 = 95.5/53.6/50.0 and reproduced EP4 = 94.4/53.8/50.0. The project reports binary anomaly metrics and cannot establish multi-type reproduction quality.

## Metric and inference comparability

- `clip_ad/metrics.py` computes positive-class binary AUROC/AP/F1-max. F1-max selects the best threshold on the evaluated category. Product macro values weight categories equally.
- `DAPO/test_multi_defect.py` computes one-vs-rest multi-type AUROC, macro AP across channels and macro F1 from argmax predictions. Its pixel AP/F1 have different targets and averaging from project binary metrics.
- `DAPO/test.py` forms each pixel anomaly map by summing abnormal channels from the full defect-vocabulary softmax, then summing feature layers.
- `clip_ad/readout.py` instead computes a two-anchor softmax using normal and the normalized average abnormal text embedding. `clip_ad/evaluate.py` uses the mean of those binary maps across layers. Summing versus averaging a fixed number of layers merely rescales scores; changing softmax anchors changes the score function and can change rankings and metrics.
- Both image paths compare CLS features against normal and normalized mean-abnormal anchors at temperature 0.07. Actual candidate vocabularies, checkpoints and dataset views in the DAPO reproduction remain unverified.
- Project AUPRO follows upstream’s 200 sampled thresholds, 8-connected regions, FPR strictly below 0.3 and normalization over the retained sampled FPR range. Matching metric code does not establish matching input predictions.

## What to resolve next

1. Recover the actual spectral + region reports. Verify checkpoint paths and saved experiment configuration; compare checkpoint identities before interpreting region results.
2. Evaluate original DAPO weights and all three experiment checkpoints through the same evaluator with identical MPDD view and vocabulary. For upstream pixel parity, additionally evaluate the sum of abnormal typed probabilities.
3. Obtain the run configs and training logs; verify region loss is active and inspect weighted objectives and spectral gates. Current JSON reports omit checkpoint identity, config, data path and sample counts.
4. Use a predefined validation protocol for checkpoint selection and repeat runs across seeds before claiming improvements.

The existing `Plan/MPDD_EXPERIMENT_REVIEW.md` refers to older, currently absent reports and quotes region image AUROC 59.16. That historical value cannot be reconciled with the current duplicated region files; it is not used as a current experiment result here.
