"""
@File         : region-inference.py
@Time         : 2026/03/02
@Author       : JiaJun Li
@Description  : Run DuckEnformer forward inference for genomic regions (one row -> one npz).
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


def setup_logging(script_name: str, proj_dir: Path):
    return get_logger(
        name=script_name,
        output_dir=proj_dir / "logs",
        log_type="Run",
        level="INFO",
    )


def safe_name(s: str) -> str:
    return (
        str(s)
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
    )


def fetch_sequence(genome: Fasta, chrom: str, start: int, end: int) -> str | None:
    """
    Fetch sequence from genome with [start, end) (0-based, end-exclusive).
    Require length == L.
    """
    if start < 0:
        return None
    try:
        seq = genome[chrom][start:end]
    except KeyError:
        return None
    if len(seq) != L:
        return None
    return seq


def main(args):
    logger = setup_logging(Path(__file__).stem, args.Proj_dir)

    # ----- IO -----
    sites_file = Path(args.sites_file)
    if not sites_file.exists():
        raise FileNotFoundError(sites_file)

    out_dir = Path(args.out_dir) / args.ckpt
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DuckEnformer region inference (one row -> one npz)")
    logger.info(f"sites_file: {sites_file}")
    logger.info(f"out_dir: {out_dir}")
    logger.info(f"L: {L}")
    logger.info("=" * 60)

    # ----- Load regions table -----
    df = pd.read_csv(sites_file)

    required_cols = ["chrom", "start", "end"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"sites_file missing columns: {missing}. Required: {required_cols}")

    # Optional dedup: per region unique
    if args.unique_regions:
        df = df.drop_duplicates(subset=["chrom", "start", "end"]).copy()

    # ----- Load genome fasta -----
    genome_fa = Path(args.genome_fa)
    genome = Fasta(genome_fa, as_raw=True, sequence_always_upper=True)
    logger.info(f"Loaded genome: {genome_fa}")

    # ----- Load model -----
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

    # ----- Run inference per region -----
    n_total = len(df)
    n_ok = 0

    for idx, row in df.iterrows():
        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])

        # optional: add prefix like "Chr"
        if args.chrom_prefix and not chrom.startswith(args.chrom_prefix):
            chrom = f"{args.chrom_prefix}{chrom}"

        seq = fetch_sequence(genome, chrom, start, end)
        if seq is None:
            logger.warning(f"[Skip] bad region or length!=L: {chrom}:{start}-{end}")
            continue

        x = one_hot_encode(seq)  # (L,4)
        x = tf.convert_to_tensor(x[None, :, :], dtype=tf.float32)  # (1, L, 4)

        pred = infer(model, x)["duck"]   # (1, 896, 240)
        preds = pred[0].numpy()          # (896, 240)

        region_id = f"Chr{chrom}_{start}_{end}"
        out_file = out_dir / f"{safe_name(region_id)}.npz"

        np.savez_compressed(
            out_file,
            chrom=np.array([chrom]),
            start=np.array([start]),
            end=np.array([end]),
            preds=preds,
        )

        n_ok += 1
        if n_ok % 50 == 0:
            logger.info(f"Progress: {n_ok:,}/{n_total:,} saved...")

    logger.info(f"Done. Saved {n_ok:,}/{n_total:,} npz files to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuckEnformer region inference (one row -> one npz)")

    parser.add_argument("--Proj_dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint ID (e.g. 20251229_145341)")

    parser.add_argument("--genome_fa", type=Path, required=True,
                        help="Path to reference genome FASTA (not distributed in this repo).")

    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output root directory (defaults to <Proj_dir>/results/inference/interest_regions)")

    parser.add_argument("--sites_file", type=Path, required=True,
                        help="CSV containing columns: chrom,start,end (0-based, end-exclusive)")

    parser.add_argument("--chrom_prefix", type=str, default="",
                        help="Optional chromosome prefix, e.g. 'Chr'. If set and chrom doesn't start with it, prefix will be added.")

    parser.add_argument("--unique_regions", action="store_true",
                        help="If set, deduplicate by (chrom,start,end)")

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.Proj_dir / "results" / "inference" / "interest_regions"
    main(args)