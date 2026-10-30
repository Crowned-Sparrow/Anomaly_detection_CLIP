#!/bin/bash
# Submit from the login node; SLURM runs this same script on a compute node.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run_cluster.sh [--dry-run] {train|evaluate|test} [Python arguments...]

Examples (run on the cluster login node):
  bash scripts/run_cluster.sh train --data-root /home/$USER/datasets/mvtec --epochs 1
  bash scripts/run_cluster.sh train --config configs/baseline.json --data-root /datasets/mvtec
  bash scripts/run_cluster.sh evaluate --checkpoint runs/clip-ad-train-123/epoch_4.pth \
      --dataset visa --data-root /datasets/visa
  bash scripts/run_cluster.sh test

Environment overrides:
  CLUSTER_CONDA_ENV  Existing Conda environment name or path (default: clip-ad)
  CLUSTER_PARTITION  Your permitted partition (default: student9696)
  CLUSTER_QOS        QoS (default: qos9696 on student9696; empty to use account default)
  CLUSTER_GPU_GRES   Full SLURM GRES (default: one 96 GB MIG on student9696)
  CLUSTER_CPUS       CPUs per task (default: 4)
  CLUSTER_MEM        Job RAM (default: 16G)
  CLUSTER_TIME       Wall time (default: 01:00:00)
  CLUSTER_CUDA_MODULE CUDA module (default: cuda/13.0)

Training defaults to configs/spectral.json. Training/evaluation outputs default
to runs/<job-name>-<job-id>[/evaluation.json]. Explicit Python arguments override
these defaults. Relative paths are resolved from the project root.
EOF
}

batch=false
dry_run=false
case "${1:-}" in
    --help|-h|'') usage; exit 0 ;;
    --batch) batch=true; shift ;;
    --dry-run) dry_run=true; shift ;;
esac
mode=${1:-}
case "$mode" in
    train|evaluate|test) shift ;;
    *) usage >&2; exit 2 ;;
esac

if ! "$batch"; then
    project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
    export CLIP_AD_PROJECT_ROOT="$project_root"
    partition=${CLUSTER_PARTITION:-student9696}
    default_qos=
    case "$partition" in
        student9696)
            default_gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition_4g.96gb:1
            default_qos=qos9696
            ;;
        *) default_gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition_2g.48gb:1 ;;
    esac
    submission=(sbatch --export=ALL --chdir="$project_root"
        --job-name="clip-ad-$mode" --partition="$partition"
        --nodes=1 --ntasks=1 --cpus-per-task="${CLUSTER_CPUS:-4}"
        --mem="${CLUSTER_MEM:-16G}" --time="${CLUSTER_TIME:-01:00:00}"
        --gres="${CLUSTER_GPU_GRES:-$default_gres}"
        --output="$project_root/logs/%x-%j.out"
        --error="$project_root/logs/%x-%j.err")
    qos=${CLUSTER_QOS-$default_qos}
    if [[ -n $qos ]]; then
        submission+=(--qos="$qos")
    fi
    submission+=("$project_root/scripts/run_cluster.sh" --batch "$mode" "$@")
    if "$dry_run"; then
        printf '%q ' "${submission[@]}"
        printf '\n'
        exit 0
    fi
    command -v sbatch >/dev/null || { echo 'sbatch is unavailable; submit from the cluster login node.' >&2; exit 1; }
    # SLURM opens log files before the batch script starts.
    mkdir -p -- "$project_root/logs"
    exec "${submission[@]}"
fi

: "${SLURM_JOB_ID:?The batch stage must run inside a SLURM allocation}"
: "${CLIP_AD_PROJECT_ROOT:?Missing project root; submit using this script}"
cd -- "$CLIP_AD_PROJECT_ROOT"

# Lmod normally initializes in the site shell profile; initialize it if needed.
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/lmod.sh ]]; then
    source /etc/profile.d/lmod.sh
fi
type module >/dev/null 2>&1 || { echo 'Lmod module command is unavailable.' >&2; exit 1; }
module load miniconda3
module load "${CLUSTER_CUDA_MODULE:-cuda/13.0}"
# The cluster guide requires source activate for non-interactive shells.
# Conda activation scripts may reference unset variables.
set +u
source activate "${CLUSTER_CONDA_ENV:-clip-ad}"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Preserve CUDA_VISIBLE_DEVICES set by SLURM for the allocated MIG.
output="runs/${SLURM_JOB_NAME:-clip-ad-$mode}-${SLURM_JOB_ID}"
case "$mode" in
    train) command_args=(python -m clip_ad.train --config configs/spectral.json --output "$output" "$@") ;;
    evaluate) command_args=(python -m clip_ad.evaluate --output "$output/evaluation.json" "$@") ;;
    test) command_args=(python -m unittest discover -s tests -v "$@") ;;
esac
printf 'Job %s on %s; Conda environment: %s\n' "$SLURM_JOB_ID" "${SLURM_NODELIST:-unknown}" "${CLUSTER_CONDA_ENV:-clip-ad}"
printf 'Running: '
printf '%q ' "${command_args[@]}"
printf '\n'
# sbatch already placed this single-task batch script on the allocated node.
# Run directly, as in the guide's batch example; srun may be absent here.
exec "${command_args[@]}"
