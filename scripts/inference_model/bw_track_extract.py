"""
@File         :   bw_track_extract.py
@Time         :   2026/03/02, 14:46
@Author       :   JiaJun Li
@Description  :   Extract signal from a single-track bigWig for a given genomic region,
optionally bin to 128bp and crop to match DuckEnformer output bins.

Typical use (match Enformer 896 bins):
  - L=196608, bin_bp=128 => 1536 bins
  - crop_bins=320 => 896 bins

Usage examples:

1) Centered window (recommended for Enformer sync)
python bw_track_extract.py \
  --bw /path/to/track_51.bw \
  --track_id 51 \
  --chrom 1 --center 1005000 \
  --chrom_prefix Chr \
  --out out_track51_chr1_1005000.npz

2) Explicit interval (must be length L if you want binning/cropping exactly)
python bw_track_extract.py \
  --bw /path/to/track_51.bw \
  --track_id 51 \
  --chrom 1 --start 1000000 --end 1196608 \
  --out out.npz

3) No save (just logger summary)
python bw_track_extract.py \
  --bw /path/to/track_51.bw \
  --track_id 51 \
  --chrom 1 --center 1005000 \
  --no-save
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyBigWig
from typing import Optional

from utils.logger_util import get_logger


def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Run",
                      level="INFO")


def safe_name(s: str) -> str:
    return (
        str(s)
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
    )


def build_region(
        chrom: str,
        center: Optional[int],
        start: Optional[int],
        end: Optional[int],
        l: int,
):
    """
    Return (chrom, start, end, center_pos)
    Using 0-based start, end-exclusive convention.
    """
    if center is not None:
        half = l // 2
        start = int(center) - half
        end = int(center) + half
        center_pos = int(center)
    else:
        if start is None or end is None:
            raise ValueError("Either provide --center or both --start and --end")
        start = int(start)
        end = int(end)
        center_pos = (start + end) // 2
    if start < 0:
        raise ValueError(f"start < 0 ({start}). Move center/start away from chromosome boundary.")
    return chrom, start, end, center_pos


def read_bw_values(bw_path: Path, chrom: str, start: int, end: int) -> np.ndarray:
    bw = pyBigWig.open(str(bw_path))
    try:
        sig = bw.values(chrom, start, end, numpy=True)  # length = end-start
    finally:
        bw.close()
    sig = np.nan_to_num(sig, nan=0.0)
    return sig.astype(np.float32)


def bin_average(signal_bp: np.ndarray, bin_bp: int) -> np.ndarray:
    """
    Average signal into bins of size bin_bp.
    Requires len(signal_bp) divisible by bin_bp.
    """
    L = len(signal_bp)
    if L % bin_bp != 0:
        raise ValueError(f"Length {L} not divisible by bin_bp={bin_bp}")
    n_bins = L // bin_bp
    return signal_bp.reshape(n_bins, bin_bp).mean(axis=1).astype(np.float32)


def crop_center_bins(binned: np.ndarray, crop_bins: int) -> np.ndarray:
    """
    Crop equal bins from both ends.
    E.g. 1536 bins with crop_bins=320 -> 896 bins
    """
    if crop_bins <= 0:
        return binned
    if 2 * crop_bins >= len(binned):
        raise ValueError(f"crop_bins too large: {crop_bins} for n_bins={len(binned)}")
    return binned[crop_bins: len(binned) - crop_bins]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--Proj_dir", type=Path, default=Path(__file__).resolve().parents[2], help="Project root directory")
    ap.add_argument("--track_term", type=str, default="A-Cortex-PK29", help="Track term name")

    ap.add_argument("--chrom", type=str, required=True, help="Chrom name or number (e.g. 1 or Chr1)")
    ap.add_argument("--chrom_prefix", type=str, default="", help="If set, prefix will be added when chrom doesn't start with it (e.g. 'Chr')")

    # region
    ap.add_argument("--center", type=int, default=None, help="Center position (bp, 0-based or 1-based? use same as bw convention; usually 0-based in pyBigWig coords)")
    ap.add_argument("--start", type=int, default=None, help="Start (0-based)")
    ap.add_argument("--end", type=int, default=None, help="End (0-based, exclusive)")

    # sync options
    ap.add_argument("--L", type=int, default=196_608, help="Window length (default 196608)")
    ap.add_argument("--bin_bp", type=int, default=128, help="Bin size for averaging (default 128)")
    ap.add_argument("--no-bin", action="store_true", help="Do not bin; keep bp-resolution signal")
    ap.add_argument("--crop_bins", type=int, default=320, help="Crop bins on both ends after binning (default 320 -> 1536->896). Set 0 to disable.")

    # output
    ap.add_argument("--out", type=Path, default=None, help="Output npz path. Default auto name in current dir.")
    ap.add_argument("--no-save", action="store_true", help="Do not save npz; only print summary.")

    args = ap.parse_args()

    logger = setup_logging(Path(__file__).stem, args.Proj_dir)

    logger.info(args)

    chrom = str(args.chrom)
    if args.chrom_prefix and not chrom.startswith(args.chrom_prefix):
        chrom = f"{args.chrom_prefix}{chrom}"

    chrom, start, end, center_pos = build_region(
        chrom=chrom,
        center=args.center,
        start=args.start,
        end=args.end,
        l=args.L,
    )

    length = end - start
    logger.info(f"[INFO] region: {chrom}:{start}-{end} (len={length}) center={center_pos}")

    # bw files
    bw_track_map = args.Proj_dir / "data" / "processed" / "enformerTgts" / "targets_duck.txt"
    track_term = args.track_term
    bw_track_df = pd.read_csv(bw_track_map, sep="\t", header=0)
    match = bw_track_df[bw_track_df["identifier"] == track_term]

    if match.empty:
        raise ValueError(f"track_term '{track_term}' not found in {bw_track_map}")

    row = match.iloc[0]

    track_id = int(row["index"])
    bw_file = row["file"]

    logger.info(f"[INFO] bw: {bw_file} | track_id={track_id}")

    sig_bp = read_bw_values(bw_file, chrom, start, end)

    # x-axis in kb (centered)
    x_bp = np.arange(start, end)
    x_kb = (x_bp - center_pos) / 1000.0

    if args.no_bin:
        signal = sig_bp
        x = x_kb
        mode = "bp"
    else:
        if length != args.L:
            raise ValueError(f"When binning, region length must equal L={args.L}. Got len={length}. Use --center or set start/end to match L.")
        binned = bin_average(sig_bp, args.bin_bp)  # 196608/128=1536
        binned = crop_center_bins(binned, args.crop_bins)

        # Make matching x for bins
        n_bins_full = args.L // args.bin_bp
        x_kb_binned = x_kb.reshape(n_bins_full, args.bin_bp).mean(axis=1)
        x_kb_binned = crop_center_bins(x_kb_binned, args.crop_bins)

        signal = binned
        x = x_kb_binned
        mode = f"bin{args.bin_bp}_crop{args.crop_bins}"

    logger.info(f"[OK] extracted signal shape: {signal.shape} | mode={mode}")
    logger.info(f"[OK] signal summary: min={signal.min():.4f}, max={signal.max():.4f}, mean={signal.mean():.4f}")

    if args.no_save:
        return

    if args.out is None:
        region_id = f"{chrom}_{start}_{end}"
        out_name = f"track{track_id}__{safe_name(region_id)}.npz"
        out_path = args.Proj_dir / "results" / "inference" / "interest_regions" / out_name
    else:
        out_path = args.out

    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        track_id=np.array([track_id]),
        chrom=np.array([chrom]),
        start=np.array([start]),
        end=np.array([end]),
        center=np.array([center_pos]),
        mode=np.array([mode]),
        x_kb=x.astype(np.float32),
        signal=signal.astype(np.float32),
        bw_path=np.array([str(bw_file)]),
    )

    logger.info(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
