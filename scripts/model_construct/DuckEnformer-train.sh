#!/bin/bash
# ==========================================================
# DuckEnformer training launcher
# Author: JiaJun Li
# Description:
#   - Activate conda environment
#   - Switch to CUDA 11.0 explicitly
#   - Launch training with logging
# ==========================================================

set -euo pipefail

# --------------------------
# Basic paths & env (override via environment variables as needed)
# --------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/software/anaconda3}"
CONDA_ENV="${CONDA_ENV:-tf241_py38}"

TRAIN_SCRIPT="./train.py"

LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.0}"

# --------------------------
# Prepare log directory
# --------------------------
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "[INFO] DuckEnformer training launcher"
echo "[INFO] Start time: $(date)"
echo "============================================"

# --------------------------
# Init conda (non-interactive)
# --------------------------
if [ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] conda.sh not found at ${CONDA_ROOT}"
    exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "[INFO] Conda env activated: ${CONDA_ENV}"
echo "[INFO] Python: $(which python)"

# --------------------------
# Switch to CUDA 11.0
# --------------------------
if [ ! -d "${CUDA_HOME}" ]; then
    echo "[ERROR] CUDA_HOME not found: ${CUDA_HOME}"
    exit 1
fi

export CUDA_HOME="${CUDA_HOME}"

# Make sure LD_LIBRARY_PATH is defined
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Remove CUDA 12.6 if present
export PATH=$(echo "$PATH" | sed 's#:/usr/local/cuda-12.6/bin##g')
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's#:/usr/local/cuda-12.6/lib64##g')

# Add CUDA 11.0
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

echo "[INFO] >>> Switched to CUDA 11.0"
echo "[INFO] CUDA_HOME=${CUDA_HOME}"

# --------------------------
# GPU runtime control
# --------------------------
export CUDA_VISIBLE_DEVICES=3
export TF_FORCE_GPU_ALLOW_GROWTH=true

# --------------------------
# Go to project
# --------------------------
cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot cd to ${PROJECT_ROOT}"
    exit 1
}

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] Train script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# --------------------------
# Sanity check (optional but useful)
# --------------------------
echo "[INFO] GPU status:"
nvidia-smi || true

python - <<EOF || true
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
EOF

# --------------------------
# Run training
# --------------------------
echo "--------------------------------------------"
echo "[INFO] Launching training..."
echo "[INFO] Log file: ${LOG_FILE}"
echo "--------------------------------------------"

python "${TRAIN_SCRIPT}" \
    --Proj_dir "${PROJECT_ROOT}" \
    2>&1 | tee "${LOG_FILE}"

echo "--------------------------------------------"
echo "[INFO] Training finished"
echo "[INFO] End time: $(date)"
echo "[INFO] Log saved to: ${LOG_FILE}"
echo "============================================"