#!/bin/bash

# ====== Conda 激活 + 开始计时 ======
START_TIME=$(date +%s)

source ~/.bashrc
conda activate deepbioinfo

echo ">>> Running bamCoverage..."
echo "Start time: $(date)"

# ====== 路径设置 ======
WORKDIR="/storage-06/lijj/work_dir/DuckEnformer"
BAM="/storage-05/workspace/epi_duck/dedup_bam/PK32-DNao-H3K4ME3_deduplicate.bam"
OUTDIR="${WORKDIR}/data/epigenome/bw"
mkdir -p "$OUTDIR"

# ====== 调用 bamCoverage ======
BAM_BASE=$(basename "$BAM")
PREFIX=${BAM_BASE%%_deduplicate.bam}
OUTBW="${OUTDIR}/${PREFIX}.bw"

echo "Input BAM:  $BAM"
echo "Output BW:  $OUTBW"

bamCoverage \
  -b "$BAM" \
  -o "$OUTBW" \
  --binSize 1 \
  --normalizeUsing RPGC \
  --effectiveGenomeSize 1211992756 \
  --extendReads \
  -p 16

# ====== 结束计时 ======
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "Done! Output: $OUTBW"
echo "Elapsed time: ${RUNTIME} seconds ($(echo "$RUNTIME / 60" | bc -l) minutes)"
echo "Finish time: $(date)"