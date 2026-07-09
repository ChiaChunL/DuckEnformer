"""
@File         :   delta-inference.py
@Time         :   2026/01/20, 20:22
@Author       :   JiaJun Li
@Description  :   Predict delta effect for custom variant(s) using DuckEnformer.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from pyfaidx import Fasta
from tqdm import tqdm

from models.enformer import Enformer, one_hot_encode
from engine.inference import infer

from utils.logger_util import get_logger
from utils.load_data import load_yaml

L = 196_608
CENTER = L // 2


def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Run",
                      level="INFO")


def parse_variants(variant_str: str):
    """
    Parse variant string like:
    "5:54133530:A:T;1:626577:A:C"
    """
    variants = []
    for item in variant_str.split(";"):
        parts = item.strip().split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid variant format: {item}")
        chrom, pos, ref, alt = parts
        variants.append({
            "Chr": chrom,
            "Pos": int(pos),
            "Ref": ref.upper(),
            "Alt": alt.upper(),
            "variant_key": item.strip(),
        })
    return variants


def fetch_sequence(genome, chrom, pos):
    """
    Fetch L bp sequence centered at pos (1-based).
    Return None if out of boundary.
    """
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

    logger.info("=" * 60)
    logger.info("DuckEnformer custom variant inference")
    logger.info(args)
    logger.info("=" * 60)

    # Load genome fasta
    genome = Fasta(args.genome_fa, as_raw=True, sequence_always_upper=True)
    logger.info(f"Loaded genome: {args.genome_fa}")

    # Parse variants
    variants = parse_variants(args.sites)
    logger.info(f"Parsed {len(variants)} variant(s)")

    # Checkpoint config
    cfg_path = args.Proj_dir / "checkpoints" / "duck_enformer" / args.ckpt / "config.yaml"
    cfg = load_yaml(cfg_path)
    ckpt_path = cfg_path.parent / "best"
    logger.info(f"Loaded config: {cfg_path}")

    # Build model
    model = Enformer(
        channels=cfg["model"]["channels"],
        num_heads=cfg["model"]["num_heads"],
        num_transformer_layers=cfg["model"]["num_transformer_layers"],
        pooling_type=cfg["model"]["pooling_type"],
    )

    # Restore checkpoint

    ckpt = tf.train.Checkpoint(model=model)
    best_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if best_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")
    ckpt.restore(best_ckpt).expect_partial()
    logger.info(f"Checkpoint restored from: {ckpt_path}")

    # warm-up
    _ = infer(model, tf.zeros((1, L, 4), dtype=tf.float32))

    # Build sequences
    seq_refs = []
    seq_alts = []
    variant_keys = []

    for v in variants:
        seq_ref = fetch_sequence(genome, v["Chr"], v["Pos"])
        if seq_ref is None:
            logger.warning(f"Skip {v['variant_key']}: out of boundary")
            continue

        # check ref allele
        ref_base = seq_ref[CENTER]
        if ref_base != v["Ref"]:
            logger.warning(
                f"Ref mismatch at {v['variant_key']}: "
                f"genome={ref_base}, input={v['Ref']}"
            )
            continue

        seq_alt = seq_ref[:CENTER] + v["Alt"] + seq_ref[CENTER + 1:]

        seq_refs.append(one_hot_encode(seq_ref))
        seq_alts.append(one_hot_encode(seq_alt))
        variant_keys.append(v["variant_key"])

    if len(seq_refs) == 0:
        raise RuntimeError("No valid variants left after filtering")

    # Inference
    x = tf.convert_to_tensor(
        np.stack(seq_refs + seq_alts),
        dtype=tf.float32,
    )

    pred = infer(model, x)["duck"]
    pred_ref, pred_alt = tf.split(pred, 2, axis=0)
    delta = pred_alt - pred_ref

    # Save result
    out_file = args.out_dir / args.ckpt / f"custom_variants_{'_'.join(args.sites.split(';'))}.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_file,
        variant_key=np.array(variant_keys),
        pred_ref=pred_ref.numpy(),
        pred_alt=pred_alt.numpy(),
        delta=delta.numpy(),
    )

    logger.info(f"Saved result to: {out_file}")
    logger.info("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Site of interest delta effect inference")

    parser.add_argument(
        "--Proj_dir",
        type=Path,
        default=Path.cwd(),
        help="Project root directory",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint ID (e.g. 20251229_145341)",
    )

    parser.add_argument(
        "--genome_fa",
        type=Path,
        required=True,
        help="Path to reference genome FASTA (not distributed in this repo).",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for results (defaults to <Proj_dir>/results/inference/interest_variant)",
    )

    parser.add_argument(
        "--sites",
        type=str,
        default="16:15065999:C:T",
        help='Variant string, e.g. "5:54133530:A:T;1:626577:A:C"',
    )

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.Proj_dir / "results" / "inference" / "interest_variant"
    main(args)
