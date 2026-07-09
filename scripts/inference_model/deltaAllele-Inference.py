"""
@File         :   deltaAllele-Inference.py
@Time         :   2026/01/17, 14:10
@Author       :   JiaJun Li
@Description  :   
"""

import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm

from models.enformer import Enformer, one_hot_encode
from engine.inference import infer, flush_batch
from utils.logger_util import get_logger
from utils.load_data import load_yaml

L = 196_608
CENTER = L // 2


def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Inference",
                      level="INFO")


def main(args):
    logger = setup_logging(Path(__file__).stem, args.Proj_dir)

    logger.info("=" * 60)
    logger.info("Starting model Inference")
    logger.info(args)
    logger.info("=" * 60)

    # Load BED file
    bed_path = args.Proj_dir / "data" / "processed" / "duckASI_l197k" / "Duck_ASI_variants_197k-ATAC-H3K27AC.bed"
    bed = pd.read_csv(bed_path, sep="\t", header=None, names=["Chr", "start", "end", "variant_key"])

    # Load genome fasta
    genome = Fasta(args.genome_fa, as_raw=True, sequence_always_upper=True)

    # Output dir
    # args.out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint config
    cfg_path = args.Proj_dir / "checkpoints" / "duck_enformer" / args.ckpt / "config.yaml"
    cfg = load_yaml(cfg_path)
    logger.info(f"Loaded config: {cfg_path}")

    # Build model
    # model = Enformer(
    #     channels=cfg["model"]["channels"],
    #     num_heads=cfg["model"]["num_heads"],
    #     num_transformer_layers=cfg["model"]["num_transformer_layers"],
    #     pooling_type=cfg["model"]["pooling_type"],
    # )

    # # Restore checkpoint
    ckpt_path = cfg_path.parent / "best"

    # ckpt = tf.train.Checkpoint(model=model)
    # best_ckpt = tf.train.latest_checkpoint(ckpt_path)
    # if best_ckpt is None:
    #     raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")
    # ckpt.restore(best_ckpt).expect_partial()
    # logger.info(f"Checkpoint restored from: {ckpt_path}")
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"Using {strategy.num_replicas_in_sync} GPUs")

    with strategy.scope():
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
    logger.info("Model instantiated")

    # Run Inference
    logger.info("Running model inference ...")

    # warm-up (important for tf.function)
    dummy = tf.zeros((1, L, 4), dtype=tf.float32)
    _ = infer(model, dummy)

    BATCH_SIZE = args.batch_size
    batch_ref, batch_alt, batch_keys = [], [], []
    batch_id = 0

    # # for test
    N = bed.shape[0]
    half = N // 2
    if args.chunk == 0:
        bed = bed.iloc[:half]
    elif args.chunk == 1:
        bed = bed.iloc[half:]

    out_dir = args.out_dir / args.ckpt / str(args.chunk)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(bed.itertuples(index=False), total=len(bed), desc="Inference", dynamic_ncols=True):
        # 1) reference sequence
        seq_ref = genome[row.Chr][row.start:row.end]
        if len(seq_ref) != L:
            continue

        # 2) alt sequence（中心 1bp 替换）
        # alt 碱基从 variant_key 里取，避免再次算 Pos
        # variant_key: Chr:Pos:Ref:Alt
        try:
            parts = row.variant_key.split(":")
            if len(parts) != 4:
                logger.warning(f"Invalid variant_key: {row.variant_key}")
                continue
            alt = parts[3]
        except Exception:
            continue

        seq_alt = seq_ref[:CENTER] + alt + seq_ref[CENTER + 1:]

        # 3) one-hot
        batch_ref.append(one_hot_encode(seq_ref))
        batch_alt.append(one_hot_encode(seq_alt))
        batch_keys.append(row.variant_key)

        # 4) flush
        if len(batch_ref) == BATCH_SIZE:
            flush_batch(model, batch_ref, batch_alt, batch_keys, batch_id, out_dir)
            batch_id += 1

    # flush last
    flush_batch(model, batch_ref, batch_alt, batch_keys, batch_id, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script description")

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
        help="Run ID for model checkpoint, e.g. 20251229_145341",
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
        help="Output directory for results (defaults to <Proj_dir>/results/inference)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Infer batch size",
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Chunk number",
    )
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.Proj_dir / "results" / "inference"
    main(args)
