#!/bin/bash
#SBATCH -J DuckEnformer
#SBATCH -p YuanMingYuan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem=256G
#SBATCH --open-mode=append
# NOTE: SBATCH -o/-e paths are relative to the directory `sbatch` is run from
# (expected to be the repo root). Adjust if you submit from elsewhere.
#SBATCH -o logs/_slurm/DuckEnformerTrain_output_%j.out
#SBATCH -e logs/_slurm/DuckEnformerTrain_error_%j.err


set -euo pipefail

echo "Visible GPUs:"
nvidia-smi
# ==========================================
# Script to run DuckEnformer training launcher on Slurm
# ==========================================
# Author: Jiajun Li
# Email: JIAJUNLI@cau.edu.cn
# Created: 2025-12
# Description:
#   This script runs the DuckEnformer training launcher on Slurm.
#   - Activate conda environment
#   - Switch to CUDA 11.0 explicitly
#   - Launch training with logging
# ==========================================

# =========================================================
# Slurm runtime info
# =========================================================
echo "============================================"
echo "[INFO] SLURM_JOB_ID        = $SLURM_JOB_ID"
echo "[INFO] SLURM_NODELIST     = $SLURM_NODELIST"
echo "[INFO] SLURM_GPUS=${SLURM_GPUS:-N/A}"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] Job start time     = $(date)"
echo "============================================"

# =========================================================
# Runtime environment variables
# =========================================================
export TF_FORCE_GPU_ALLOW_GROWTH=true

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=4

# =========================================================
# Basic paths & configuration (override via environment variables as needed)
# =========================================================
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/software/anaconda3}"
CONDA_ENV="${CONDA_ENV:-tf241_py38}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.0}"

TRAIN_SCRIPT="./train.py"

LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "[INFO] DuckEnformer training launcher"
echo "[INFO] Start time: $(date)"
echo "============================================"

# =========================================================
# Conda initialization
# =========================================================
if [ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] conda.sh not found at ${CONDA_ROOT}"
    exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "[INFO] Conda env activated: ${CONDA_ENV}"
echo "[INFO] Python path: $(which python)"

# =========================================================
# CUDA 11.0 environment setup
# =========================================================
if [ ! -d "${CUDA_HOME}" ]; then
    echo "[ERROR] CUDA_HOME not found: ${CUDA_HOME}"
    exit 1
fi

export CUDA_HOME="${CUDA_HOME}"

# Remove CUDA 12.6 if present
export PATH=$(echo "$PATH" | sed 's#:/usr/local/cuda-12.6/bin##g')
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's#:/usr/local/cuda-12.6/lib64##g')

# Add CUDA 11.0
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

echo "[INFO] >>> Switched to CUDA 11.0"
echo "[INFO] CUDA_HOME=${CUDA_HOME}"

# =========================================================
# Go to project directory
# =========================================================
cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot cd to ${PROJECT_ROOT}"
    exit 1
}

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] Train script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# =========================================================
# Sanity checks (non-fatal)
# =========================================================
echo "--------------------------------------------"
echo "[INFO] GPU status:"
nvidia-smi || true

python - <<EOF || true
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
EOF
echo "--------------------------------------------"


# =========================================================
# Run training
# =========================================================
echo "[INFO] Launching DuckEnformer training..."
echo "[INFO] Training log: ${LOG_FILE}"
echo "--------------------------------------------"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${TRAIN_SCRIPT}" \
    --Proj_dir "${PROJECT_ROOT}" \
    2>&1 | tee "${LOG_FILE}"

echo "--------------------------------------------"
echo "[INFO] Training finished"
echo "[INFO] End time: $(date)"
echo "[INFO] Log saved to: ${LOG_FILE}"
echo "============================================"