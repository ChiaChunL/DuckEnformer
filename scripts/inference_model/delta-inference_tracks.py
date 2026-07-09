"""
@File         :   delta-inference_from_file.py
@Time         :   2026/03/02
@Author       :   JiaJun Li
@Description  :   Read variants from a CSV (variant_key, TrackID, ...),
                  run DuckEnformer inference per row, and save ONE npz per row.
                  File name: track{TrackID}__{variant_key}.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from pyfaidx import Fasta

from models.enformer import Enformer, one_hot_encode
from engine.inference import infer
from utils.logger_util import get_logger
from utils.load_data import load_yaml

L = 196_608
CENTER = L // 2


def setup_logging(script_name: str, proj_dir: Path):
    return get_logger(
        name=script_name,
        output_dir=proj_dir / "logs",
        log_type="Run",
        level="INFO",
    )


def safe_name(s: str) -> str:
    """Make string safe for filenames."""
    return (
        str(s)
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
    )


def parse_variant_key(vkey: str):
    """Parse 'chr:pos:ref:alt' -> (chrom, pos, ref, alt)"""
    parts = str(vkey).strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid variant_key format: {vkey}")
    chrom, pos, ref, alt = parts
    return chrom, int(pos), ref.upper(), alt.upper()


def fetch_sequence(genome, chrom: str, pos: int):
    """Fetch L bp sequence centered at pos (1-based)."""
    start = pos - CENTER - 1
    end = pos + CENTER - 1
    if start < 0:
        return None
    seq = genome[chrom][start:end]
    if len(seq) != L:
        return None
    return seq


def main(args):
    logger = setup_logging(Path(__file__).stem, args.Proj_dir)

    sites_file = Path(args.sites_file)
    if not sites_file.exists():
        raise FileNotFoundError(sites_file)

    out_dir = Path(args.out_dir) / args.ckpt
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DuckEnformer inference from sites_file (one row -> one npz)")
    logger.info(f"sites_file: {sites_file}")
    logger.info(f"out_dir: {out_dir}")
    logger.info("=" * 60)

    # Load table
    df = pd.read_csv(sites_file)
    if "variant_key" not in df.columns:
        raise ValueError("sites_file must contain column: variant_key")
    if "TrackID" not in df.columns:
        raise ValueError("sites_file must contain column: TrackID")

    # Optional: 每个 variant_key 只跑一次（不按行跑）
    if args.unique_variants:
        df = df.drop_duplicates(subset=["variant_key"]).copy()

    # Load genome fasta
    genome = Fasta(args.genome_fa, as_raw=True, sequence_always_upper=True)
    logger.info(f"Loaded genome: {args.genome_fa}")

    # Load model config + checkpoint
    cfg_path = args.Proj_dir / "checkpoints" / "duck_enformer" / args.ckpt / "config.yaml"
    cfg = load_yaml(cfg_path)
    ckpt_path = cfg_path.parent / "best"
    logger.info(f"Loaded config: {cfg_path}")

    model = Enformer(
        channels=cfg["model"]["channels"],
        num_heads=cfg["model"]["num_heads"],
        num_transformer_layers=cfg["model"]["num_transformer_layers"],
        pooling_type=cfg["model"]["pooling_type"],
    )

    ckpt = tf.train.Checkpoint(model=model)
    best_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if best_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")
    ckpt.restore(best_ckpt).expect_partial()
    logger.info(f"Checkpoint restored from: {ckpt_path}")

    # warm-up
    _ = infer(model, tf.zeros((1, L, 4), dtype=tf.float32))

    n_total = len(df)
    n_ok = 0

    for i, row in df.iterrows():
        vkey = row["variant_key"]
        track_id = int(row["TrackID"])

        track_name = row["TrackName"] if "TrackName" in df.columns else ""
        tissue = row["Tissue"] if "Tissue" in df.columns else ""

        try:
            chrom, pos, ref, alt = parse_variant_key(vkey)
        except Exception as e:
            logger.warning(f"[Skip] bad variant_key: {vkey} | {e}")
            continue

        seq_ref = fetch_sequence(genome, chrom, pos)
        if seq_ref is None:
            logger.warning(f"[Skip] out of boundary: {vkey}")
            continue

        # Check ref base (center)
        ref_base = seq_ref[CENTER]
        if ref_base != ref:
            logger.warning(f"[Skip] ref mismatch: {vkey} | genome={ref_base}, input={ref}")
            continue

        seq_alt = seq_ref[:CENTER] + alt + seq_ref[CENTER + 1:]

        x_ref = one_hot_encode(seq_ref)
        x_alt = one_hot_encode(seq_alt)

        x = tf.convert_to_tensor(np.stack([x_ref, x_alt]), dtype=tf.float32)  # (2, L, 4)
        pred = infer(model, x)["duck"]                                       # (2, 896, 240)

        pred_ref = pred[0].numpy()   # (896, 240)
        pred_alt = pred[1].numpy()   # (896, 240)
        delta = pred_alt - pred_ref  # (896, 240)

        # File name: track{TrackID}__{variant_key}.npz
        out_file = out_dir / f"track{track_id}__{safe_name(vkey)}.npz"

        np.savez(
            out_file,
            variant_key=np.array([str(vkey)]),
            TrackID=np.array([track_id]),
            TrackName=np.array([str(track_name)]),
            Tissue=np.array([str(tissue)]),
            pred_ref=pred_ref,
            pred_alt=pred_alt,
            delta=delta,
        )

        n_ok += 1
        if n_ok % 50 == 0:
            logger.info(f"Progress: {n_ok:,}/{n_total:,} saved...")

    logger.info(f"Done. Saved {n_ok:,}/{n_total:,} npz files to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuckEnformer inference from CSV (one row -> one npz)")

    parser.add_argument("--Proj_dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint ID (e.g. 20251229_145341)")
    parser.add_argument("--genome_fa", type=Path, required=True,
                        help="Path to reference genome FASTA (not distributed in this repo).")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory (defaults to <Proj_dir>/results/inference/interest_variant)")

    parser.add_argument("--sites_file", type=Path, required=True,
                        help="CSV containing at least columns: variant_key, TrackID (and optionally TrackName/Tissue)")

    parser.add_argument("--unique_variants", action="store_true",
                        help="If set, only run one inference per unique variant_key (ignore TrackID duplicates)")

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.Proj_dir / "results" / "inference" / "interest_variant"
    main(args)