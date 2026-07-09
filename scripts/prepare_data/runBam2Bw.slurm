#!/bin/bash
#SBATCH -J BAM2BW
#SBATCH -p YuanMingYuan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=high
#SBATCH --mem=256G
#SBATCH --open-mode=append
# NOTE: SBATCH -o/-e paths are relative to the directory `sbatch` is run from
# (expected to be the repo root). Adjust if you submit from elsewhere.
#SBATCH -o logs/_slurm/BAM2BW_output_%j.out
#SBATCH -e logs/_slurm/BAM2BW_error_%j.err

set -euo pipefail

# ==========================================
# Script to run bam files to bw files  on Slurm
# ==========================================
# Author: Jiajun Li
# Email: JIAJUNLI@cau.edu.cn
# Created: 2025-12
# Description:
#   Convert ATAC-seq / CUT&Tag BAM files to bigWig tracks.
#   - Activate conda environment
#   - Ensure BAM index (.bai)
#   - Generate bigWig with bamCoverage
# ==========================================

# =========================================================
# Slurm runtime info
# =========================================================
echo "============================================"
echo "[INFO] SLURM_JOB_ID        = $SLURM_JOB_ID"
echo "[INFO] SLURM_NODELIST     = $SLURM_NODELIST"
echo "[INFO] Job start time     = $(date)"
echo "============================================"


# =========================================================
# Basic paths & configuration (override via environment variables as needed)
# =========================================================
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/software/anaconda3}"
CONDA_ENV="${CONDA_ENV:-duckenformer}"
# Directory containing raw ATAC-seq/CUT&Tag BAM files (not part of this repo)
EPIGEN_DIR="${EPIGEN_DIR:?Set EPIGEN_DIR to your local Epigenomes/Duck directory}"

SCRIPT="scripts/prepare_data/DuckEpigenomics.bam2bw.py"

LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/bam2bw_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "[INFO] BAM2BW launcher"
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
# Go to project directory
# =========================================================
cd "${PROJECT_ROOT}" || {
    echo "[ERROR] Cannot cd to ${PROJECT_ROOT}"
    exit 1
}

if [ ! -f "${SCRIPT}" ]; then
    echo "[ERROR] Train script not found: ${SCRIPT}"
    exit 1
fi
# =========================================================
# Run convert
# =========================================================
echo "[INFO] Launching BAM2BW ..."
echo "[INFO] Log: ${LOG_FILE}"
echo "--------------------------------------------"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${SCRIPT}" \
    --Proj_dir "${PROJECT_ROOT}" \
    --Epigen_dir "${EPIGEN_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "--------------------------------------------"
echo "[INFO] Convert finished"
echo "[INFO] End time: $(date)"
echo "[INFO] Log saved to: ${LOG_FILE}"
echo "============================================"