"""
@File         :   train.py
@Time         :   2025/12/19, 10:58
@Author       :   JiaJun Li
@Description  :   DuckEnformer training entry
"""

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import sonnet as snt
import yaml

from models import enformer
from datasets.dataset import get_dataset
from engine.evaluate import evaluate_model
from engine.trainer import Trainer
from utils.logger_util import get_logger


def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Train",
                      level="INFO")


def setup_gpu(gpu_id=None, memory_growth=True, logger=None):
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        if logger:
            logger.warning("No GPU detected, using CPU")
        return

    if gpu_id is not None:
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        if logger:
            logger.info(f"Using GPU:{gpu_id}")

    if memory_growth:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

def main(args):
    script_name = Path(__file__).stem
    logger = setup_logging(script_name, args.Proj_dir)
    setup_gpu(gpu_id=0, logger=logger)

    logger.info("===================================")
    logger.info(args)
    logger.info("Training started")
    logger.info("===================================")

    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")

    # =====================
    # Load config
    # =====================
    config_path = args.Proj_dir / "configs" / "duck.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg_str = yaml.dump(cfg, sort_keys=False)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]

    logger.info(f"Loaded config from: {config_path.resolve()}")
    logger.info(f"Config hash: {cfg_hash}")
    logger.info("===== Experiment Config =====")
    logger.info("\n" + cfg_str)
    logger.info("=============================")


    # =====================
    # Dataset
    # =====================
    batch_size = int(cfg["train"]["batch_size"])
    shuffle_buffer = int(cfg["data"].get("shuffle_buffer", 1024))

    train_ds = (
        get_dataset(cfg["data"]["dataset_name"], "train")
        .shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        get_dataset(cfg["data"]["dataset_name"], "valid")
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # =====================
    # Model
    # =====================
    model = enformer.Enformer(
        channels=cfg['model']['channels'],
        num_heads=cfg['model']['num_heads'],
        num_transformer_layers=cfg['model']['num_transformer_layers'],
        pooling_type=cfg['model']['pooling_type'],
    )

    # =====================
    # Optimizer
    # =====================
    lr = tf.Variable(0.0, trainable=False)
    optimizer = snt.optimizers.Adam(lr)

    # =====================
    # Trainer
    # =====================
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_var=lr,
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=cfg,
        proj_dir=args.Proj_dir,
        run_id=run_id,
        logger=logger,
        config_path=config_path,
        resume=args.resume,
    )

    trainer.train(evaluator_fn=evaluate_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--Proj_dir',
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root directory (defaults to this script's directory).",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None
    )
    args = parser.parse_args()

    main(args)