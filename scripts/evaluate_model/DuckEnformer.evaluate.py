"""
@File         :   DuckEnformer/src/engine/evaluator.py
@Time         :   2025/12/28, 10:10
@Author       :   JiaJun Li
@Description  :   Evaluate Enformer model on a dataset split.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd
from datasets.dataset import get_dataset
from models.enformer import Enformer
from engine.evaluator import evaluate_model
from utils.logger_util import get_logger
from utils.load_data import load_yaml, load_json

# ======================
# Main
# ======================
def main(args: argparse.Namespace):
    logger = get_logger(
        name=Path(__file__).stem,
        output_dir=args.Proj_dir / "logs",
        log_type="Eval",
        level="INFO",
    )

    logger.info("=" * 60)
    logger.info("Starting model evaluation")
    logger.info(f"Args: {vars(args)}")
    logger.info("=" * 60)

    # --------------------------------------------------
    # Checkpoint config
    # --------------------------------------------------
    cfg_path = args.Proj_dir / "checkpoints" / "duck_enformer" / args.ckpt / "config.yaml"
    cfg = load_yaml(cfg_path)
    logger.info(f"Loaded config: {cfg_path}")

    # --------------------------------------------------
    # Load dataset statistics
    # --------------------------------------------------
    stats_path = (
            Path(cfg["data"]["dataset_root"])
            / cfg["data"]["dataset_name"]
            / "statistics.json"
    )
    stats = load_json(stats_path)

    num_targets = int(stats["num_targets"])
    num_seqs = {
        "train": int(stats["train_seqs"]),
        "valid": int(stats["valid_seqs"]),
        "test": int(stats["test_seqs"]),
    }
    logger.info(f"Dataset: {cfg['data']['dataset_name']}")
    logger.info(f"Num targets: {num_targets}")
    logger.info(f"Num sequences: {num_seqs}")

    # --------------------------------------------------
    # Build dataset
    # --------------------------------------------------
    dataset = (
        get_dataset(cfg["data"]["dataset_name"], args.split)
        .batch(int(cfg["train"]["batch_size"]))
        .prefetch(tf.data.AUTOTUNE)
    )
    eval_dataset_nums = num_seqs.get(args.split, num_seqs["test"])
    logger.info(f"Loaded {args.split} dataset: {eval_dataset_nums} sequences")

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    model = Enformer(
        channels=cfg["model"]["channels"],
        num_heads=cfg["model"]["num_heads"],
        num_transformer_layers=cfg["model"]["num_transformer_layers"],
        pooling_type=cfg["model"]["pooling_type"],
    )
    logger.info("Model instantiated")

    # --------------------------------------------------
    # Restore checkpoint
    # --------------------------------------------------
    ckpt_path = cfg_path.parent / "best"
    ckpt = tf.train.Checkpoint(model=model)
    best_ckpt = tf.train.latest_checkpoint(ckpt_path)
    ckpt.restore(best_ckpt).expect_partial()
    logger.info(f"Checkpoint restored from: {ckpt_path}")

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------
    logger.info("Running PearsonR evaluation...")
    pearson_per_target = evaluate_model(
        model=model,
        dataset=dataset,
        head=cfg["model"]["head"],
        max_steps=eval_dataset_nums // cfg["train"]["batch_size"],
    )

    pearson_np = pearson_per_target.numpy()
    mean_pearson = float(pearson_np.mean())

    logger.info(f"Mean PearsonR = {mean_pearson:.4f}")

    # --------------------------------------------------
    # Visualize prediction vs true signal tracks
    # --------------------------------------------------
    logger.info("Running visual evaluation (prediction vs true tracks)...")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    out_dir = (args.out_dir / args.ckpt) or (args.Proj_dir / "results" / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"pearson_{args.split}.json"
    out_npy = out_dir / f"pearson_{args.split}.npy"
    out_csv = out_dir / f"pearson_{args.split}.csv"

    with open(out_json, "w") as f:
        json.dump(
            {
                "mean": mean_pearson,
                "per_target": pearson_np.tolist(),
            },
            f,
            indent=2,
        )

    np.save(out_npy, pearson_np)

    df = pd.DataFrame({
        "target_index": np.arange(len(pearson_np)),
        "pearson": pearson_np,
    })

    df.to_csv(out_csv, index=False, sep=",")

    logger.info(f"Saved PearsonR to:")
    logger.info(f"  JSON: {out_json}")
    logger.info(f"  NPY : {out_npy}")
    logger.info(f"  CSV : {out_csv}")

    logger.info("Evaluation finished ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DuckEnformer model")

    parser.add_argument(
        "--Proj_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Run ID for model checkpoint: 20251226_121503",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for results (defaults to <Proj_dir>/results/model_evaluation)",
    )

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.Proj_dir / "results" / "model_evaluation"
    main(args)
