# CLIP spatial-spectral anomaly detection

Compose DAPO's existing blocks with optional new modules. Develop locally; install
dependencies, run numerical tests, train and evaluate on the server.

## Structure

```text
DAPO/                       unchanged upstream reference, commit 290411d
clip_ad/
  model.py                  composed model, factory and block checkpoint interface
  blocks/spectral.py        image windows → radial FFT → MLP → gated residual
  blocks/region_pooling.py   training-only GT-masked pooling of patch embeddings
  readout.py                shared image/text and dense patch/text similarity
  losses.py                 independent objectives and weighted composition
  data.py                   reuse DAPO's metadata dataset
  train.py                  server training
  evaluate.py               server binary evaluation
configs/
  baseline.json             DAPO blocks and original loss weights
  spectral.json             same experiment with spectral fusion enabled
  spectral_region.json      spectral fusion + optional region–text alignment loss
tests/                      synthetic component and composition checks
```

The model reuses `DAPO.adapter.LinearAdapter`, `LearnablePrompt`,
`VisualPromptTuning`, and DAPO's vendored OpenCLIP. Losses reuse `FocalLoss` and
`BinaryDiceLoss`. Upstream source files are not edited or copied. DAPO/OpenCLIP
imports are selected explicitly in `clip_ad/vendor.py`; do not import another
installed `open_clip` package first.

```text
image + visual prompts → frozen CLIP → patch adapters ──────────────┐
image → patch-aligned windows → radial FFT → spectral MLP ──────────┤
                                                   optional fusion
candidate names → learned prompts → frozen CLIP text → text anchors
                                     ↓
            shared similarity readout → per-layer maps + CLS logits
                                     ↓ training only
         independent segmentation and classification objectives
```

## Server workflow

### SLURM cluster launcher

[`scripts/run_cluster.sh`](scripts/run_cluster.sh) follows the supplied
`20260923_Cluster_User_Guide.pdf`. Connect to NetBird and SSH to the login node
first. Transfer the complete repository including `DAPO/`, then use an existing
Conda environment with the project dependencies installed. The guide recommends
Conda packages first, then conda-forge, with pip only as a fallback.

```bash
# On the login node, select your prepared environment.
export CLUSTER_CONDA_ENV=my_dapo_env

# Preview submission without scheduling a job.
bash scripts/run_cluster.sh --dry-run train --data-root /home/$USER/datasets/mvtec --epochs 1

# Run numerical tests, then submit a spectral smoke test.
bash scripts/run_cluster.sh test
bash scripts/run_cluster.sh train --data-root /home/$USER/datasets/mvtec --epochs 1

# Full baseline and spectral runs are separate jobs.
bash scripts/run_cluster.sh train --config configs/baseline.json --data-root /home/$USER/datasets/mvtec
bash scripts/run_cluster.sh train --data-root /home/$USER/datasets/mvtec

# Once the spectral training job finishes, substitute its job ID below.
bash scripts/run_cluster.sh evaluate --checkpoint runs/clip-ad-train-12345/epoch_4.pth \
  --dataset visa --data-root /home/$USER/datasets/visa

# Use only a partition you have permission to access; this selects a 96 GB MIG.
CLUSTER_PARTITION=student9696 CLUSTER_MEM=32G CLUSTER_TIME=08:00:00 \
  bash scripts/run_cluster.sh train --data-root /home/$USER/datasets/mvtec
```

The launcher defaults to `student4848`, one 48 GB MIG, 4 CPUs, 16 GB RAM,
and one hour. Adjust the resource settings for full runs. It loads `miniconda3`
and `cuda/13.0`, then uses `source activate` inside the job. The guide states
that CUDA 12.6 is incompatible with these GPUs; use a compatible PyTorch build
in your environment as well. The launcher preserves SLURM's GPU visibility.
Run `bash scripts/run_cluster.sh --help` for all resource overrides.

Logs appear in `logs/clip-ad-<mode>-<job-id>.out` and `.err`. Outputs default to
`runs/clip-ad-<mode>-<job-id>`; evaluation writes `evaluation.json` there.
Pass `--output` to choose another location. Use `squeue -u "$USER"` to monitor,
`tail -f logs/clip-ad-train-12345.out` to follow training, and `scancel 12345`
to cancel a job. Training and tests run directly inside the compute-node batch allocation.
Datasets can remain under `/home`; `/compute_home` is optional shared scratch
for write-intensive work and requires cleanup after use. No data is copied by
the launcher. Prepare model weights in advance if compute nodes lack internet
access, as described below.

### Direct commands inside a compute allocation

Transfer the complete project, including `DAPO/`, to the server. Use the same
working environment and dataset preparation as your successful DAPO reproduction.
All commands below run from this project's root, not from `DAPO/`.

```bash
python -m unittest discover -s tests -v

# Evaluate your existing DAPO weights through the composed spatial model.
python -m clip_ad.evaluate --checkpoint /checkpoints/dapo/epoch_4.pth \
  --config configs/baseline.json --dataset visa --data-root /datasets/visa \
  --output runs/baseline/visa.json

# Verify the composed spatial baseline first.
python -m clip_ad.train --config configs/baseline.json \
  --data-root /datasets/mvtec --output runs/baseline

# First spectral smoke test; use a new directory for the full run.
python -m clip_ad.train --config configs/spectral.json \
  --data-root /datasets/mvtec --output runs/spectral_smoke --epochs 1

python -m clip_ad.train --config configs/spectral.json \
  --data-root /datasets/mvtec --output runs/spectral

python -m clip_ad.evaluate --checkpoint runs/spectral/epoch_4.pth \
  --dataset visa --data-root /datasets/visa --output runs/spectral/visa.json
```

The source MVTec experiment intentionally uses DAPO's `meta.json` **test** partition
for supervised source training, as upstream does. The held-out target dataset must
be separate. The provided prompt vocabulary, alias-to-channel mapping, loss weights,
Adam settings and seeds are taken from upstream training. Review the JSON configs
if your reproduced experiment used different settings. Mask preprocessing preserves
upstream training's integer conversion before thresholding, including its treatment
of fractional mask boundary values.

To compare against existing DAPO weights without retraining, pass
`--init-dapo /path/to/epoch_4.pth` to training. With `--epochs 1`, this performs
one additional training epoch; it is initialization, not resume. To evaluate an
original checkpoint without training, use the evaluation command above with
`--config configs/baseline.json`. Architecture
and prompt settings must match the original run. No extra optimizer parameters are
added to the baseline objective; CLIP weights are explicitly frozen while gradients
flow through the encoders to learned prompts.

Resume a new-format run with `--resume runs/spectral/epoch_0.pth`, the same config,
and a total `--epochs` greater than the saved epoch. Checkpoints save learned blocks,
configuration, vocabulary, optimizer, epoch and RNG states. Frozen CLIP weights
are reconstructed from `model.pretrained`; they are not duplicated in checkpoints.
For offline servers, provide a locally cached or local-path pretrained checkpoint.

## Adding blocks and objectives

`SpatialSpectralCLIP` owns feature extraction and prediction and receives no GT.
`fusion=None` gives the spatial baseline. The optional `SpectralFusion` receives
already-adapted patch features and the preprocessed image. It does not subclass
or modify DAPO's adapter. Gates start at zero, preserving the corresponding
spatial model's initial predictions. The spectral encoder starts receiving
gradients once the gates move away from zero.

`LossComposer` accepts named `nn.Module` objectives and matching weights. Each
objective implements `forward(outputs, targets)`. Existing defaults reproduce
`4 × sum_layers(Focal + Dice_normal + Dice_anomaly) + CE_image`. Add region losses
as independent objectives consuming `patch_embeddings` and `text_embeddings`.
The baseline and spectral configurations leave region alignment disabled.
`configs/spectral_region.json` enables it with weight 0.1 and temperature 0.07;
these are initial experiment settings, not validated optimal values.

`MaskedRegionPooling` uses area coverage of source GT masks at the patch grid.
It pools one feature per image and positive defect type, combining disconnected
parts of the same type. Background regions are excluded, and small defects retain
fractional coverage. `RegionTextAlignmentLoss` normalizes each pooled feature and
uses cosine-softmax cross-entropy against all text anchors, including normal as
a negative candidate. It averages over regions and then feature layers. Empty
defect batches contribute a differentiable zero. Gradients reach patch features
and learned text prompts; GT masks never enter inference. No additional learned
parameters or checkpoint blocks are introduced. This objective does not implement
UniSpector's angular-margin loss, and no query decoder is enabled.

After the existing baseline and spectral checks, submit the additional experiment:

```bash
bash scripts/run_cluster.sh test
bash scripts/run_cluster.sh train --config configs/spectral_region.json \
  --data-root /path/to/mvtec --epochs 1
```

Use the same held-out dataset and vocabulary when comparing all three configs.
The epoch metrics include `region_alignment` for enabled runs. Omit its weight
or set it to zero to skip the new objective completely. Start the region experiment
as a separate run: `--resume` requires an identical experiment configuration and
cannot enable a new loss.

New checkpoints keep visual prompts, text prompts, spatial adapters and spectral
fusion as separate blocks. Inference reconstructs the same model from saved config.
Use `--defect-names /path/to/names.json` during evaluation for a JSON list of unseen
candidate names; channel zero is always normal and the rest follow that list.
Template tensors are rebuilt for the new names while learned context tokens transfer.

## Evaluation and validation limits

The supplied evaluation entry point reports per-product and macro image/pixel
AUROC and AP. Classes lacking both labels have null metrics. It uses mean per-layer
binary similarity probabilities for localization and binary CLS logits for image
scores. It does not reproduce all upstream reporting or multi-type metrics;
the model also exposes dense `type_probabilities` for a typed evaluator. Compare
both baseline and spectral models through this same evaluator. Changing the
candidate vocabulary changes the averaged abnormal text anchor, so hold it fixed
when comparing experiments.

The spectral path uses grayscale, mean-centered Hann windows, float32 FFT log
magnitudes, radial averaging and a shared MLP. Windows align to the square patch
grid with replicated boundaries; defaults are 28-pixel windows, stride 14 and
16 bins at image size 518. Image dimensions must be divisible by the patch grid;
window size must be at least patch size. A parity mismatch shifts centers by half
a pixel. Standard OpenAI CLIP image normalization is assumed. Radial descriptors
discard phase and orientation. The CLS path is unchanged, although joint training
can indirectly affect its learned prompts.

Local validation covers syntax and source integration only: PyTorch is not installed
locally. Numerical tests and end-to-end baseline equivalence must run on the server.
