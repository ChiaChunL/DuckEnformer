#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_region_corrs.py
- Iterate DuckEnformer region prediction npz files (preds: [896,240])
- For each region, extract experimental signal from 240 bigWigs on same region
  (bin 128bp + crop 320 => 896 bins)
- Compute correlation per track, save summary table for filtering

Assumptions:
- region npz filename like: Chr5_54032595_54229203.npz   (chrom_start_end)
  or use npz meta if you have (optional)
- targets file: targets_duck.txt has columns: index, identifier, file, ...
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyBigWig
from typing import Optional, Tuple

from utils.logger_util import get_logger

# Enformer region setup
L = 196_608
BIN_BP = 128
CROP_BINS = 320


def setup_logging(script_name: str, proj_dir: Path):
    return get_logger(
        name=script_name,
        output_dir=proj_dir / "logs",
        log_type="Run",
        level="INFO",
    )


def parse_region_from_name(p: Path) -> Tuple[str, int, int]:
    """
    Parse chrom/start/end from filename like:
      Chr5_54032595_54229203.npz
      5_54032595_54229203.npz
    """
    name = p.stem
    m = re.match(r"^(Chr)?(\w+)[_\-](\d+)[_\-](\d+)$", name)
    if not m:
        raise ValueError(f"Cannot parse region from filename: {p.name}")
    chrom = m.group(2)
    start = int(m.group(3))
    end = int(m.group(4))
    # standardize chrom to 'ChrX' (match your bw usage)
    if str(chrom).startswith("Chr"):
        chrom = chrom.replace("Chr", "")
    return chrom, start, end


def read_bw_values(bw_path: str, chrom: str, start: int, end: int) -> np.ndarray:
    bw = pyBigWig.open(bw_path)
    try:
        sig = bw.values(chrom, start, end, numpy=True)
    finally:
        bw.close()
    sig = np.nan_to_num(sig, nan=0.0).astype(np.float32)
    return sig


def bin_average(signal_bp: np.ndarray, bin_bp: int) -> np.ndarray:
    n = len(signal_bp)
    if n % bin_bp != 0:
        raise ValueError(f"Length {n} not divisible by bin_bp={bin_bp}")
    return signal_bp.reshape(n // bin_bp, bin_bp).mean(axis=1).astype(np.float32)


def crop_center_bins(arr: np.ndarray, crop_bins: int) -> np.ndarray:
    if crop_bins <= 0:
        return arr
    if 2 * crop_bins >= len(arr):
        raise ValueError(f"crop_bins too large: {crop_bins} for n={len(arr)}")
    return arr[crop_bins: len(arr) - crop_bins]


def corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fast pearson for 1D arrays; return nan if variance is zero.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    vx = np.sqrt((x * x).mean())
    vy = np.sqrt((y * y).mean())
    if vx == 0 or vy == 0:
        return float("nan")
    return float((x * y).mean() / (vx * vy))


def corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman via rank transform (no scipy dependency).
    """
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return corr_pearson(rx, ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Proj_dir", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--pred_dir", type=Path, required=True,
                    help="Directory containing region prediction npz (preds: [896,240])")
    ap.add_argument("--targets", type=Path, default=None,
                    help="targets_duck.txt (index, identifier, file, ...); "
                         "defaults to <Proj_dir>/data/processed/enformerTgts/targets_duck.txt")
    ap.add_argument("--out_csv", type=Path, required=True, help="Output summary CSV (long table)")
    ap.add_argument("--max_regions", type=int, default=0, help="For debug. 0 means all.")
    ap.add_argument("--only_prefix", type=str, default="Chr", help="Only process files whose name startswith this prefix")
    ap.add_argument("--spearman", action="store_true", help="Also compute spearman (slower)")
    args = ap.parse_args()
    if args.targets is None:
        args.targets = args.Proj_dir / "data" / "processed" / "enformerTgts" / "targets_duck.txt"

    logger = setup_logging(Path(__file__).stem, args.Proj_dir)
    logger.info(args)

    # load target mapping
    tgt = pd.read_csv(args.targets, sep="\t")
    if "index" not in tgt.columns or "file" not in tgt.columns or "identifier" not in tgt.columns:
        raise ValueError("targets file must contain columns: index, identifier, file")

    # ensure sorted by track index 0..239
    tgt = tgt.sort_values("index").reset_index(drop=True)
    bw_files = tgt["file"].astype(str).tolist()
    track_ids = tgt["index"].astype(int).tolist()
    track_terms = tgt["identifier"].astype(str).tolist()

    logger.info(f"[OK] loaded targets: {len(tgt)} tracks")

    pred_files = sorted(args.pred_dir.glob("*.npz"))
    if args.only_prefix:
        pred_files = [p for p in pred_files if p.name.startswith(args.only_prefix)]
    if args.max_regions and args.max_regions > 0:
        pred_files = pred_files[: args.max_regions]

    logger.info(f"[OK] region npz files: {len(pred_files)}")

    rows = []
    for ridx, p in enumerate(pred_files, 1):
        try:
            chrom, start, end = parse_region_from_name(p)
        except Exception as e:
            logger.warning(f"[Skip] {p.name}: {e}")
            continue

        # Load preds
        dd = np.load(p)
        if "preds" not in dd.files:
            logger.warning(f"[Skip] {p.name}: no 'preds' key")
            continue
        preds = dd["preds"]  # (896,240)
        if preds.shape[0] != 896:
            logger.warning(f"[Skip] {p.name}: preds shape {preds.shape} (expect 896,240)")
            continue

        # Fetch experimental per track
        # Note: region length must equal L
        if (end - start) != L:
            logger.warning(f"[Skip] {p.name}: region len {end-start} != L={L}")
            continue

        # compute per track
        # We'll read each bigwig and compute corr with preds[:,j]
        for j in range(len(bw_files)):
            bw_path = bw_files[j]
            try:
                sig_bp = read_bw_values(bw_path, chrom, start, end)      # (196608,)
                sig_bin = bin_average(sig_bp, BIN_BP)                   # (1536,)
                sig_896 = crop_center_bins(sig_bin, CROP_BINS)           # (896,)
            except Exception as e:
                # bigwig missing chrom etc.
                rows.append({
                    "region_file": p.name,
                    "chrom": chrom, "start": start, "end": end,
                    "track_index": track_ids[j],
                    "track_term": track_terms[j],
                    "bw_file": bw_path,
                    "pearson_r": np.nan,
                    "spearman_rho": np.nan if args.spearman else None,
                    "error": str(e),
                })
                continue

            pred_curve = preds[:, j].astype(np.float32)

            r = corr_pearson(pred_curve, sig_896)
            rho = corr_spearman(pred_curve, sig_896) if args.spearman else None

            rows.append({
                "region_file": p.name,
                "chrom": chrom, "start": start, "end": end,
                "track_index": track_ids[j],
                "track_term": track_terms[j],
                "bw_file": bw_path,
                "pearson_r": r,
                "spearman_rho": rho,
                "error": "",
            })

        if ridx % 5 == 0:
            logger.info(f"Processed {ridx}/{len(pred_files)} regions...")

    out_df = pd.DataFrame(rows)

    # drop spearman column if not requested (keeps csv clean)
    if not args.spearman and "spearman_rho" in out_df.columns:
        out_df = out_df.drop(columns=["spearman_rho"])

    out_df.to_csv(args.out_csv, index=False)
    logger.info(f"[OK] wrote: {args.out_csv}")
    logger.info("Done.")


if __name__ == "__main__":
    main()