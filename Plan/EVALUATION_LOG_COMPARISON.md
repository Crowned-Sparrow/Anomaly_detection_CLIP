# Evaluation log comparison

All 58 reports lack the corrected evaluator protocol marker and use the source MVTec vocabulary. Treat these as legacy results. Region denotes spectral + region. Epoch numbers are zero-based; epoch_0 is after the first training epoch. MAD-Sim region epochs 3 and 4 are absent.

Comparisons use macro metrics. Vocabulary and category sets match within each dataset; reports do not establish checkpoint or sample identity. Best-over-epoch results are retrospective test-set summaries, not a validation-based checkpoint selection rule.

## visa

### Epochs exceeding baseline at the same epoch

| Metric | Spectral | Spectral + region |
|---|---|---|
| Image AUROC | 1, 2, 3 | 2, 3, 4 |
| Image AP | 1, 2, 3 | 2, 3, 4 |
| Image F1-max | 1, 2, 3, 4 | 2, 3, 4 |
| Pixel AUROC | 1 | 0, 1, 2 |
| Pixel AP | 1 | 1, 3, 4 |
| Pixel F1-max | 1, 2 | 1, 2, 4 |
| Pixel AUPRO | 0, 2 | 0, 1, 2 |

### Best available epoch for each metric

All values are percentages. Deltas are percentage points relative to the best baseline value for that metric. Each cell may select a different checkpoint.

| Metric | Baseline | Spectral | Spectral + region |
|---|---|---|---|
| Image AUROC | 84.7778 (epoch 0) | 84.9704 (epoch 1); delta +0.1926 | 84.7307 (epoch 2); delta -0.0471 |
| Image AP | 86.8836 (epoch 0) | 85.7759 (epoch 1); delta -1.1078 | 85.5378 (epoch 2); delta -1.3458 |
| Image F1-max | 80.0468 (epoch 0) | 80.0954 (epoch 1); delta +0.0485 | 79.8890 (epoch 2); delta -0.1579 |
| Pixel AUROC | 91.6239 (epoch 1) | 91.7432 (epoch 1); delta +0.1192 | 92.4946 (epoch 0); delta +0.8707 |
| Pixel AP | 19.1463 (epoch 3) | 19.1632 (epoch 1); delta +0.0169 | 19.7595 (epoch 1); delta +0.6132 |
| Pixel F1-max | 26.0239 (epoch 3) | 26.6817 (epoch 1); delta +0.6578 | 26.7202 (epoch 1); delta +0.6963 |
| Pixel AUPRO | 72.6453 (epoch 3) | 76.4618 (epoch 0); delta +3.8164 | 81.4104 (epoch 0); delta +8.7650 |

## mpdd

### Epochs exceeding baseline at the same epoch

| Metric | Spectral | Spectral + region |
|---|---|---|
| Image AUROC | 2, 4 | None |
| Image AP | 0, 2, 4 | None |
| Image F1-max | 0, 4 | None |
| Pixel AUROC | 0, 2, 3, 4 | 0 |
| Pixel AP | 0, 1, 2 | 0, 1 |
| Pixel F1-max | 0, 1, 2 | 1 |
| Pixel AUPRO | 0, 2 | 0, 1, 2 |

### Best available epoch for each metric

All values are percentages. Deltas are percentage points relative to the best baseline value for that metric. Each cell may select a different checkpoint.

| Metric | Baseline | Spectral | Spectral + region |
|---|---|---|---|
| Image AUROC | 80.4742 (epoch 0) | 80.3678 (epoch 0); delta -0.1064 | 75.1734 (epoch 0); delta -5.3008 |
| Image AP | 81.9824 (epoch 3) | 85.7879 (epoch 0); delta +3.8055 | 81.0205 (epoch 0); delta -0.9619 |
| Image F1-max | 84.2278 (epoch 0) | 86.8258 (epoch 0); delta +2.5980 | 83.2059 (epoch 3); delta -1.0218 |
| Pixel AUROC | 94.5295 (epoch 3) | 95.1161 (epoch 0); delta +0.5866 | 93.9026 (epoch 1); delta -0.6269 |
| Pixel AP | 29.0219 (epoch 3) | 29.0197 (epoch 0); delta -0.0022 | 27.9386 (epoch 0); delta -1.0833 |
| Pixel F1-max | 33.0309 (epoch 0) | 33.3753 (epoch 0); delta +0.3445 | 31.0590 (epoch 1); delta -1.9719 |
| Pixel AUPRO | 71.7959 (epoch 4) | 77.6478 (epoch 0); delta +5.8520 | 80.0090 (epoch 0); delta +8.2131 |

## mad_real

### Epochs exceeding baseline at the same epoch

| Metric | Spectral | Spectral + region |
|---|---|---|
| Image AUROC | 4 | 2, 4 |
| Image AP | 4 | 4 |
| Image F1-max | None | 1, 2, 4 |
| Pixel AUROC | 0, 1, 3 | 0, 1 |
| Pixel AP | None | 1, 2, 4 |
| Pixel F1-max | None | 1, 2, 4 |
| Pixel AUPRO | 0 | 0, 1 |

### Best available epoch for each metric

All values are percentages. Deltas are percentage points relative to the best baseline value for that metric. Each cell may select a different checkpoint.

| Metric | Baseline | Spectral | Spectral + region |
|---|---|---|---|
| Image AUROC | 69.1764 (epoch 0) | 67.4107 (epoch 4); delta -1.7657 | 67.5688 (epoch 2); delta -1.6076 |
| Image AP | 89.5939 (epoch 0) | 89.3482 (epoch 4); delta -0.2456 | 88.6205 (epoch 2); delta -0.9733 |
| Image F1-max | 92.5912 (epoch 3) | 91.9995 (epoch 4); delta -0.5917 | 92.7844 (epoch 2); delta +0.1932 |
| Pixel AUROC | 85.9231 (epoch 3) | 86.1949 (epoch 3); delta +0.2718 | 87.5690 (epoch 1); delta +1.6460 |
| Pixel AP | 16.9757 (epoch 0) | 14.3590 (epoch 3); delta -2.6168 | 19.5718 (epoch 2); delta +2.5960 |
| Pixel F1-max | 20.3918 (epoch 0) | 18.3218 (epoch 3); delta -2.0700 | 21.8526 (epoch 2); delta +1.4608 |
| Pixel AUPRO | 60.2009 (epoch 4) | 61.5840 (epoch 0); delta +1.3830 | 65.3450 (epoch 0); delta +5.1440 |

## mad_sim

### Epochs exceeding baseline at the same epoch

| Metric | Spectral | Spectral + region |
|---|---|---|
| Image AUROC | 0, 1, 2, 4 | 0, 1, 2 |
| Image AP | 0, 1, 2, 4 | 1, 2 |
| Image F1-max | 0, 1 | 0, 1, 2 |
| Pixel AUROC | 1, 2 | 0, 1, 2 |
| Pixel AP | 1, 2, 4 | 1, 2 |
| Pixel F1-max | 1, 2, 4 | 0, 1, 2 |
| Pixel AUPRO | 0 | 0, 1, 2 |

### Best available epoch for each metric

All values are percentages. Deltas are percentage points relative to the best baseline value for that metric. Each cell may select a different checkpoint.

| Metric | Baseline | Spectral | Spectral + region |
|---|---|---|---|
| Image AUROC | 55.8705 (epoch 0) | 57.0249 (epoch 0); delta +1.1544 | 58.2528 (epoch 2); delta +2.3823 |
| Image AP | 90.5543 (epoch 0) | 91.0929 (epoch 0); delta +0.5386 | 90.8944 (epoch 2); delta +0.3401 |
| Image F1-max | 94.0349 (epoch 3) | 94.0008 (epoch 2); delta -0.0341 | 94.0636 (epoch 2); delta +0.0287 |
| Pixel AUROC | 86.9573 (epoch 0) | 86.6825 (epoch 0); delta -0.2748 | 87.3165 (epoch 1); delta +0.3592 |
| Pixel AP | 3.9053 (epoch 0) | 3.7519 (epoch 0); delta -0.1534 | 3.7286 (epoch 1); delta -0.1767 |
| Pixel F1-max | 6.9637 (epoch 0) | 6.7638 (epoch 0); delta -0.1999 | 6.9917 (epoch 0); delta +0.0280 |
| Pixel AUPRO | 51.4912 (epoch 4) | 46.2663 (epoch 2); delta -5.2249 | 56.6172 (epoch 0); delta +5.1260 |

