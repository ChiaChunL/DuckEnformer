"""
@File         :   trainer.py
@Time         :   2025/12/25, 20:00
@Author       :   JiaJun Li
@Description  :   Training engine for DuckEnformer
"""

from __future__ import annotations

import json
import math
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from tqdm import tqdm

import tensorflow as tf


def center_crop_target(target, target_len):
    total = tf.shape(target)[1]
    trim = (total - target_len) // 2
    return target[:, trim:trim + target_len, :]


def create_train_step(model, optimizer, clip_norm=0.2):
    """
    Create a tf.function train step.
    """
    @tf.function
    def train_step(batch, head):
        with tf.GradientTape() as tape:
            outputs = model(batch['sequence'], is_training=True)[head]
            targets = center_crop_target(batch['target'], tf.shape(outputs)[1])
            loss = tf.reduce_mean(
                tf.keras.losses.poisson(targets, outputs)
            )

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        optimizer.apply(grads, model.trainable_variables)

        return loss

    return train_step

@dataclass
class RunState:
    run_id: str
    cfg_hash: str
    ckpt_dir: Path
    best_dir: Path


class Trainer:
    """
    Minimal but solid Trainer:
    - resume / checkpoint manager / best checkpoint
    - warmup + cosine lr
    - per-epoch validation (PearsonR)
    """

    def __init__(
        self,
        *,
        model,
        optimizer,
        lr_var: tf.Variable,
        train_ds,
        val_ds,
        cfg: Dict[str, Any],
        proj_dir: Path,
        run_id: str,
        logger,
        config_path: Optional[Path] = None,
        resume: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_var = lr_var
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg
        self.proj_dir = Path(proj_dir)
        self.run_id = run_id
        self.logger = logger
        self.config_path = config_path
        self.resume = resume

        # ---- hash config (for reproducibility)
        cfg_str = self._yaml_dump(cfg)
        self.cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]

        # ---- unpack frequently used config
        self.batch_size = int(cfg["train"]["batch_size"])
        self.num_epochs = int(cfg["train"]["num_epochs"])
        self.base_lr = float(cfg["train"]["learning_rate"])
        self.warmup_steps = int(cfg["train"]["warmup_steps"])
        self.clip_norm = float(cfg["train"]["clip_norm"])
        self.save_every_steps = cfg["train"].get("save_every_steps", None)
        self.head = cfg["model"]["head"]

        self.valid_max_steps = cfg.get("eval", {}).get("valid_max_steps", None)

        # ---- global step + checkpoint
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._setup_checkpoint()

        # ---- train step
        self.train_step = create_train_step(self.model, self.optimizer, self.clip_norm)

        # ---- best score persistence
        self.best_score_file = self.best_dir / "best_score.txt"
        if self.resume and self.best_score_file.exists():
            try:
                self.best_val = float(self.best_score_file.read_text().strip())
            except Exception:
                self.best_val = -np.inf
        else:
            self.best_val = -np.inf

        # ---- restore if resume
        if self.resume:
            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint)
                self.logger.info(f"Resumed from {self.manager.latest_checkpoint}")
            else:
                self.logger.info("Resume requested, but no checkpoint found. Training from scratch.")
        else:
            self.logger.info("Training from scratch (resume disabled)")

        # ---- persist config into ckpt dir
        if self.config_path is not None and self.config_path.exists():
            shutil.copy(self.config_path, self.ckpt_dir / "config.yaml")
            self.logger.info(f"Config copied to checkpoint dir: {self.ckpt_dir / 'config.yaml'}")

        self.logger.info(f"Config hash: {self.cfg_hash}")

    # -------------------------
    # Internal helpers
    # -------------------------
    @staticmethod
    def _yaml_dump(cfg: Dict[str, Any]) -> str:
        # avoid importing yaml here; keep string stable-ish
        # (train.py already uses yaml to load and can log full cfg)
        import yaml
        return yaml.dump(cfg, sort_keys=False)

    def _setup_checkpoint(self):
        ckpt_root = self.proj_dir / "checkpoints" / "duck_enformer"
        self.ckpt_dir = ckpt_root / self.run_id
        self.best_dir = self.ckpt_dir / "best"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Checkpoint dir: {self.ckpt_dir}")

        self.ckpt = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            global_step=self.global_step,
        )
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=10)
        self.best_manager = tf.train.CheckpointManager(self.ckpt, self.best_dir, max_to_keep=1)

    def _load_dataset_stats(self) -> Dict[str, Any]:
        data_root = Path(self.cfg["data"]["dataset_root"])
        dataset_name = self.cfg["data"]["dataset_name"]
        stats_path = data_root / dataset_name / "statistics.json"
        with open(stats_path) as f:
            stats = json.load(f)
        return stats

    def _compute_lr(self, step: int, total_steps: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps

        # cosine decay after warmup
        denom = max(1, total_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / denom
        progress = min(max(progress, 0.0), 1.0)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    # -------------------------
    # Public API
    # -------------------------
    def train(self, evaluator_fn):
        """
        evaluator_fn: typically src.engine.evaluate.evaluate_model
        """
        # stats
        stats = self._load_dataset_stats()
        num_train = int(stats["train_seqs"])
        num_valid = int(stats["valid_seqs"])

        steps_per_epoch = num_train // self.batch_size
        total_steps = self.num_epochs * steps_per_epoch

        # validation steps
        if self.valid_max_steps is not None:
            max_val_steps = int(self.valid_max_steps)
        else:
            max_val_steps = max(1, num_valid // self.batch_size)

        log_every = int(self.cfg["train"].get("log_every_steps", 500))

        for epoch in range(self.num_epochs):
            epoch_losses = []

            epoch_train_ds = self.train_ds.take(steps_per_epoch)

            for batch in tqdm(
                epoch_train_ds,
                total=steps_per_epoch,
                desc=f"Epoch {epoch}",
                dynamic_ncols=True,
                leave=True,
            ):
                self.global_step.assign_add(1)
                step = int(self.global_step.numpy())

                lr_value = self._compute_lr(step, total_steps)
                self.lr_var.assign(lr_value)

                loss = self.train_step(batch, head=self.head)
                epoch_losses.append(loss)

                if self.save_every_steps and step % int(self.save_every_steps) == 0:
                    self.manager.save(checkpoint_number=step)

                if log_every and step % log_every == 0:
                    self.logger.info(f"Epoch {epoch} | Step {step} | lr = {lr_value:.6e}")

            train_loss = float(tf.reduce_mean(epoch_losses))
            self.logger.info("Running validation (approx)...")

            val_pearson = float(
                evaluator_fn(
                    self.model,
                    self.val_ds,
                    head=self.head,
                    max_steps=max_val_steps,
                ).numpy().mean()
            )

            self.logger.info(
                f"Epoch {epoch} | train loss {train_loss:.4f} | val PearsonR {val_pearson:.4f}"
            )

            # always save latest
            self.manager.save(checkpoint_number=int(self.global_step.numpy()))

            if val_pearson > self.best_val:
                self.best_val = val_pearson
                self.best_manager.save(checkpoint_number=int(self.global_step.numpy()))
                self.best_score_file.write_text(str(self.best_val))
                self.logger.info("🔥 New BEST checkpoint saved")