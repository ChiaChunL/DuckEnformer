"""
@File         :   train-multiGPUs.py
@Time         :   2025/12/19, 10:58
@Author       :   JiaJun Li
@Description  :   DuckEnformer multi-GPU training (MirroredStrategy + Keras optimizer)
"""
import argparse
from pathlib import Path
from datetime import datetime
import shutil

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import yaml

from src.models import enformer
from src.datasets.dataset import get_dataset
from src.engine.evaluate import evaluate_model
from src.utils.logger_util import get_logger


def setup_logging(script_name: str, proj_dir: Path):
    return get_logger(
        name=script_name,
        output_dir=proj_dir / "logs",
        log_type="Train",
        level="INFO",
    )


def make_distributed_datasets(train_ds, val_ds):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return train_ds.with_options(options), val_ds.with_options(options)


def create_distributed_train_step(strategy, model, optimizer, clip_norm=0.2):
    """
    Distributed train step:
      - strategy.run(step_fn)
      - strategy.reduce(MEAN)
    """
    poisson = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def train_step(dist_batch, head):

        def step_fn(batch):
            with tf.GradientTape() as tape:
                outputs = model(batch["sequence"], is_training=True)[head]
                per_example = poisson(batch["target"], outputs)
                loss = tf.reduce_mean(per_example)

            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm)

            # Keras optimizer API (compatible with MirroredStrategy)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        per_replica_loss = strategy.run(step_fn, args=(dist_batch,))
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
        return mean_loss

    return train_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Proj_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root directory (defaults to this script's directory).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--run_id", type=str, default=None, help="Specify a run_id to resume/continue")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    logger = setup_logging(script_name, args.Proj_dir)

    # -------------------------
    # Strategy & GPU visibility
    # -------------------------
    visible_gpus = tf.config.list_physical_devices("GPU")
    logger.info(f"Visible GPUs: {visible_gpus}")

    strategy = tf.distribute.MirroredStrategy()
    nrep = strategy.num_replicas_in_sync
    logger.info(f"Using MirroredStrategy with {nrep} replicas")

    # -------------------------
    # run_id
    # -------------------------
    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("===================================")
    logger.info(args)
    logger.info(f"Run ID: {run_id}")
    logger.info("Training started")
    logger.info("===================================")

    # -------------------------
    # Load config
    # -------------------------
    config_path = args.Proj_dir / "configs" / "duck.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    organism = cfg["data"]["organism"]

    global_batch_size = int(cfg["train"]["batch_size"])
    if nrep > 1 and (global_batch_size % nrep != 0):
        raise ValueError(
            f"[ERROR] train.batch_size must be divisible by num_gpus ({nrep}). "
            f"Got batch_size={global_batch_size}. "
            f"Try {nrep*4}, {nrep*5}, {nrep*6} ..."
        )
    per_replica_batch = global_batch_size // max(1, nrep)

    steps_per_epoch = int(cfg["train"]["steps_per_epoch"])
    num_epochs = int(cfg["train"]["num_epochs"])
    base_lr = float(cfg["train"]["learning_rate"])
    warmup_steps = int(cfg["train"]["warmup_steps"])
    clip_norm = float(cfg["train"]["clip_norm"])

    valid_max_steps = cfg["eval"]["valid_max_steps"]
    valid_max_steps = None if valid_max_steps is None else int(valid_max_steps)

    save_every_steps = cfg["train"].get("save_every_steps", None)
    save_every_steps = None if save_every_steps is None else int(save_every_steps)

    shuffle_buffer = int(cfg["data"].get("shuffle_buffer", 1024))

    logger.info(
        f"Config: organism={organism}, global_batch={global_batch_size}, per_replica_batch={per_replica_batch}, "
        f"steps/epoch={steps_per_epoch}, epochs={num_epochs}, base_lr={base_lr}, warmup_steps={warmup_steps}, clip_norm={clip_norm}"
    )

    # -------------------------
    # Checkpoint dirs
    # -------------------------
    ckpt_root = args.Proj_dir / "checkpoints" / "duck_enformer"
    ckpt_dir = ckpt_root / run_id
    best_dir = ckpt_dir / "best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint dir: {ckpt_dir}")
    shutil.copy(config_path, ckpt_dir / "config.yaml")

    best_score_file = best_dir / "best_score.txt"
    if args.resume and best_score_file.exists():
        best_val = float(best_score_file.read_text())
    else:
        best_val = -np.inf

    # -------------------------
    # Datasets
    # -------------------------
    train_ds = (
        get_dataset(organism, "train")
        .shuffle(shuffle_buffer)
        .batch(global_batch_size, drop_remainder=True)  # important for mirrored static shapes
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        get_dataset(organism, "valid")
        .batch(global_batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    train_ds, val_ds = make_distributed_datasets(train_ds, val_ds)

    dist_train_ds = strategy.experimental_distribute_dataset(train_ds)
    dist_train_iter = iter(dist_train_ds)

    # -------------------------
    # Model/optim/ckpt in scope
    # -------------------------
    with strategy.scope():
        model = enformer.Enformer(
            channels=cfg["model"]["channels"],
            num_heads=cfg["model"]["num_heads"],
            num_transformer_layers=cfg["model"]["num_transformer_layers"],
            pooling_type=cfg["model"]["pooling_type"],
        )

        lr = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="learning_rate")

        # ✅ Keras optimizer (compatible with MirroredStrategy)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )

        global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, global_step=global_step, lr=lr)

    manager = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=10, checkpoint_name="ckpt")
    best_manager = tf.train.CheckpointManager(ckpt, str(best_dir), max_to_keep=1, checkpoint_name="best")

    # Restore
    if args.resume:
        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint)
            logger.info(f"Resumed from {manager.latest_checkpoint} (step={int(global_step.numpy())})")
        else:
            logger.info("Resume requested, but no checkpoint found. Training from scratch.")
    else:
        logger.info("Training from scratch (resume disabled)")

    # Train step
    train_step = create_distributed_train_step(strategy, model, optimizer, clip_norm=clip_norm)

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(num_epochs):
        epoch_losses = []

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}", dynamic_ncols=True)
        for _ in pbar:
            global_step.assign_add(1)
            step = int(global_step.numpy())

            # warmup lr
            if warmup_steps > 0:
                lr_value = min(base_lr, (step / warmup_steps) * base_lr)
            else:
                lr_value = base_lr
            lr.assign(lr_value)

            dist_batch = next(dist_train_iter)
            loss = train_step(dist_batch, head=organism)
            epoch_losses.append(loss)

            if step % 50 == 0:
                pbar.set_postfix({"loss": float(loss.numpy()), "lr": float(lr.numpy())})

            if save_every_steps and (step % save_every_steps == 0):
                manager.save(checkpoint_number=global_step)

        train_loss = float(tf.reduce_mean(epoch_losses).numpy())
        logger.info(f"Epoch {epoch} train_loss={train_loss:.6f}")

        # Validation (approx)
        logger.info("Running validation (approx)...")
        val_pearson = float(
            evaluate_model(
                model=model,
                dataset=val_ds,
                head=organism,
                max_steps=valid_max_steps,
            ).numpy().mean()
        )

        logger.info(
            f"Epoch {epoch} | step={int(global_step.numpy())} | "
            f"train loss {train_loss:.4f} | val PearsonR {val_pearson:.4f}"
        )

        manager.save(checkpoint_number=global_step)

        if val_pearson > best_val:
            best_val = val_pearson
            best_manager.save(checkpoint_number=global_step)
            best_score_file.write_text(str(best_val))
            logger.info(f"🔥 New BEST checkpoint saved (PearsonR={best_val:.4f})")

    logger.info("Training finished ✅")


if __name__ == "__main__":
    main()