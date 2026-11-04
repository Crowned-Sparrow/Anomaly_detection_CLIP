# MPDD experiment review

## Available evidence

The local artifacts are `zero_mpdd_evalution.json` (spectral only),
`zero-mpdd-region-loss-evaluation.json` (region run), and the three experiment
configs. No `metrics.jsonl`, saved run `experiment.json`, checkpoints, or SLURM
logs are present locally. Training trajectories and actual server overrides
therefore remain unverified.

| Macro metric (%) | Spectral | Spectral + region | Change (percentage points) |
|---|---:|---:|---:|
| Image AUROC | 77.23 | 59.16 | -18.07 |
| Image AP | 82.78 | 66.66 | -16.13 |
| Pixel AUROC | 93.76 | 91.92 | -1.83 |
| Pixel AP | 24.38 | 24.92 | +0.54 |

Both reports use the same six MPDD categories and the same defect vocabulary.
They do not record checkpoint identity, data paths, sample counts or epoch, so
these cannot be verified from the reports alone.

## Configuration comparison

`configs/spectral_region.json` adds only region alignment weight 0.1 and
temperature 0.07 to `configs/spectral.json`. Both otherwise use five epochs,
batch size 8, learning rate 0.001, seed 42, spectral fusion, and identical source
labels and vocabulary. `configs/baseline.json` differs from the spectral config
only by setting `model.spectral` to false.

## Code inspection

- Region pooling uses positive type masks, excludes background regions and
  preserves small defects with fractional patch coverage. Regions of the same
  type in an image are pooled together, including disconnected components.
- The region objective classifies normalized pooled features against all text
  anchors, including normal. It averages over regions and feature layers.
- Its gradients reach the shared text prompts and the visual prompt path as
  well as the patch adapter and spectral fusion. Image classification uses the
  same normal anchor and the normalized mean of abnormal text anchors.
- Thus the extra objective can change image scores despite not directly using
  CLS logits. Competing gradients or source-specific text alignment are plausible
  explanations, not established causes of the observed drop.
- Weight 0.1 alone does not tell us the region term's actual influence. Loss
  magnitudes and gradient directions matter. Existing epoch logs expose loss
  magnitudes but do not measure gradient conflict.

## Inspect existing training logs

From the project root on the server, inspect both actual training runs:

```bash
cat runs/clip-ad-train-6327/experiment.json
cat runs/clip-ad-train-6327/metrics.jsonl
```

Repeat for the region run's training job ID. Compare segmentation and
classification trends separately. For the region run, compare the weighted
contributions `4 * segmentation`, `classification`, and `0.1 * region_alignment`.
Do not compare total loss across the two objectives as though it were the same
quantity. Check finite losses and spectral gate trajectories; gate values are
not a validation metric. Even falling source losses do not establish target
generalization. Verify both evaluated checkpoints are `epoch_4.pth` and use the
same MPDD view and vocabulary.

## Spatial-only baseline

Run from the project root on the cluster login node, using the same prepared
environment as the previous runs. The launcher submits to the GPU allocation.

```bash
bash scripts/run_cluster.sh train --config configs/baseline.json --data-root "$HOME/triet-work/datasets/cde-mtad/canonical/mvtec" --device cuda --epochs 5
```

The launcher defaults to a one-hour wall time. If the previous five-epoch jobs
used a longer time limit or other resource overrides, retain those overrides.
Use the same actual training dataset root as the completed spectral runs.
After successful completion, replace `BASELINE_TRAIN_JOB_ID` below with the new
training job ID. This MPDD path is the one reported for evaluation job 6360:

```bash
bash scripts/run_cluster.sh evaluate --checkpoint runs/clip-ad-train-BASELINE_TRAIN_JOB_ID/epoch_4.pth --dataset mpdd --data-root /home/slurmdang15/triet-work/datasets/cde-mtad/views/mpdd-dapo --device cuda
```

Use the final fifth epoch for all runs, rather than selecting an epoch using
MPDD test performance. Compare the new report with both existing reports.
The same seed gives a useful initial comparison, but model construction can
consume different random draws when fusion is enabled, and a single run does
not quantify training variability. Further runs should follow a predefined
protocol without tuning on MPDD test results.

Keep the current region implementation unchanged while collecting the baseline
and training logs. No additional block is needed to perform this comparison.
