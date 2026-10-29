#!/bin/bash
#SBATCH --job-name=spectral
#SBATCH --partition=student4848
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition_2g.48gb:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/spectral-%j.out
#SBATCH --error=logs/spectral-%j.err

set -eo pipefail

module load miniconda3
module load cuda12.8
source activate "${CLUSTER_CONDA_ENV:-clip-ad}"
set -u

cd "$HOME/nguyen-dd-work/Anomaly_detection_CLIP"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Keep the GPU visibility assigned by SLURM.
# Tests must pass before training starts.
srun python -m unittest discover -s tests -v

# Default to a one-epoch smoke test; set EPOCHS=5 for full training.
srun python -m clip_ad.train \
    --config configs/spectral.json \
    --data-root "$HOME/triet-work/datasets/cde-mtad/canonical/mvtec" \
    --output "runs/spectral-${SLURM_JOB_ID}" \
    --epochs "${EPOCHS:-1}"
