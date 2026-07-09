#!/bin/bash
set -euo pipefail

# ==========================================================================
# Build TFRecord training/validation/test tensors from FASTA + BigWig tracks.
#
# This step calls basenji_data.py from Calico's basenji toolkit (Apache-2.0,
# unmodified, not vendored in this repo). Before running:
#   git clone https://github.com/calico/basenji.git <BASENJI_DIR>
# See README.md "Data preparation" section for the pinned commit/version.
#
# Configure the paths below (or export them as environment variables before
# calling this script) to match your local setup.
# ==========================================================================

WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BASENJI_DIR="${BASENJI_DIR:-$WORKDIR/basenji}"

LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/dataset_split_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

START_TIME=$(date +%s)
echo ">>> Running Dataset Split ..."
echo "Start time: $(date)"

if [ ! -d "$BASENJI_DIR" ]; then
  echo "[ERROR] basenji not found at $BASENJI_DIR"
  echo "        git clone https://github.com/calico/basenji.git $BASENJI_DIR"
  exit 1
fi

export PATH="$BASENJI_DIR/bin:$PATH"
export PYTHONPATH="$BASENJI_DIR:${PYTHONPATH:-}"

echo "[INFO] Python path: $(which python)"
python --version

cd "$WORKDIR"

# ====== INPUT Files (override via environment variables as needed) ======
FASTA_FILE="${FASTA_FILE:?Set FASTA_FILE to the reference genome FASTA path}"
TARGETS_FILE="${TARGETS_FILE:-$WORKDIR/data/processed/enformerTgts/targets_duck.txt}"
UNMAP_BED="${UNMAP_BED:-$WORKDIR/data/processed/enformerTgts/unmap_macro.bed}"

OUT_DIR="${OUT_DIR:-$WORKDIR/data/processed/duck_dataset}"

echo "================ PARAMETERS ================"
SEQ_LENGTH=196608
SAMPLE_PCT=1
STRIDE_TRAIN=0.5
STRIDE_TEST=0.5

echo "SEQ_LENGTH=$SEQ_LENGTH"
echo "SAMPLE_PCT=$SAMPLE_PCT"
echo "STRIDE_TRAIN=$STRIDE_TRAIN"
echo "STRIDE_TEST=$STRIDE_TEST"
echo "FASTA_FILE=$FASTA_FILE"
echo "TARGETS_FILE=$TARGETS_FILE"
echo "UNMAP_BED=$UNMAP_BED"
echo "OUT_DIR=$OUT_DIR"
echo "==========================================="

python "$BASENJI_DIR/bin/basenji_data.py" \
  -s $SAMPLE_PCT \
  -g "$UNMAP_BED" \
  -l $SEQ_LENGTH \
  --stride_train $STRIDE_TRAIN \
  --stride_test $STRIDE_TEST \
  --local \
  -o "$OUT_DIR" \
  -p 8 \
  -t 0.1 \
  -v 0.1 \
  -w 128 \
  "$FASTA_FILE" \
  "$TARGETS_FILE"

if [ ! -d "$OUT_DIR/tfrecords" ]; then
  echo "ERROR: basenji_data did not produce tfrecords."
  exit 1
fi

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo "===== Dataset Construction Completed ====="
echo "Output directory: $OUT_DIR"
echo "Elapsed: ${RUNTIME}s ($(awk "BEGIN {print $RUNTIME/60}") min)"
echo "Finish time: $(date)"
