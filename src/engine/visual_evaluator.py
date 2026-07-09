"""
@File         : visual_evaluator.py
@Time         : 2026/01/11
@Author       : JiaJun Li
@Description  : Visualization utilities for DuckEnformer evaluation
"""

from pathlib import Path
from typing import Tuple, Sequence, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from engine.evaluator import _predict_step
from utils.logger_util import get_logger


# =========================================================
# Collect predictions
# =========================================================
def collect_predictions_for_visualization(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    head: str,
    max_batches: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect a small number of predictions for visualization.

    Parameters
    ----------
    model : tf.keras.Model
    dataset : tf.data.Dataset
    head : str
        Output head name
    max_batches : int
        Number of batches to collect

    Returns
    -------
    y_true : np.ndarray
        Shape (N, bins, targets)
    y_pred : np.ndarray
        Shape (N, bins, targets)
    """
    y_trues = []
    y_preds = []

    for step, batch in enumerate(dataset):
        if step >= max_batches:
            break

        targets, preds = _predict_step(model, batch, head)
        y_trues.append(targets.numpy())
        y_preds.append(preds.numpy())

    y_true = tf.concat(y_trues, axis=0).numpy()
    y_pred = tf.concat(y_preds, axis=0).numpy()

    return y_true, y_pred


# =========================================================
# Plot utilities
# =========================================================
def plot_signal_track(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_idx: int,
    sample_idx: int = 0,
    title: Optional[str] = None,
    out_path: Optional[Path] = None,
):
    """
    Plot predicted vs true signal track for one target.

    Parameters
    ----------
    y_true : np.ndarray
        (N, bins, targets)
    y_pred : np.ndarray
        (N, bins, targets)
    target_idx : int
        Target index to visualize
    sample_idx : int
        Which sequence/sample to visualize
    title : str, optional
    out_path : Path, optional
    """
    true_signal = y_true[sample_idx, :, target_idx]
    pred_signal = y_pred[sample_idx, :, target_idx]

    plt.figure(figsize=(10, 3))
    plt.plot(true_signal, label="True", linewidth=1)
    plt.plot(pred_signal, label="Predicted", linewidth=1)

    plt.xlabel("Genomic bins")
    plt.ylabel("Signal")
    plt.legend()

    if title:
        plt.title(title)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # PNG（位图，快速查看）
        plt.savefig(out_path.with_suffix(".png"),
                    dpi=300,
                    bbox_inches="tight")

        # PDF（矢量，论文用）
        plt.savefig(out_path.with_suffix(".pdf"),
                    bbox_inches="tight")

    plt.close()


def plot_multiple_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_indices: Sequence[int],
    sample_idx: int = 0,
    out_dir: Optional[Path] = None,
    prefix: str = "target",
):
    """
    Plot multiple targets (one figure per target).

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    target_indices : list[int]
    sample_idx : int
    out_dir : Path
    prefix : str
    """
    for t in target_indices:
        out_path = None
        if out_dir is not None:
            out_path = out_dir / f"{prefix}_{t}.png"

        plot_signal_track(
            y_true=y_true,
            y_pred=y_pred,
            target_idx=t,
            sample_idx=sample_idx,
            title=f"{prefix.capitalize()} target {t}",
            out_path=out_path,
        )


# =========================================================
# High / Mid / Low Pearson helper
# =========================================================
def select_representative_targets(
    pearson_per_target: np.ndarray,
    k: int = 3,
):
    """
    Select high / mid / low PearsonR targets.

    Returns
    -------
    dict with keys: high, mid, low
    """
    pearson = pearson_per_target

    high = np.argsort(pearson)[-k:]
    low = np.argsort(pearson)[:k]
    mid = np.argsort(np.abs(pearson - pearson.mean()))[:k]

    return {
        "high": high,
        "mid": mid,
        "low": low,
    }


# =========================================================
# One-shot visualization pipeline
# =========================================================
def run_visual_evaluation(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    head: str,
    pearson_per_target: np.ndarray,
    out_dir: Path,
    max_batches: int = 1,
    sample_idx: int = 0,
):
    """
    Full visualization pipeline:
    1. Collect predictions
    2. Select high/mid/low Pearson targets
    3. Plot signal tracks

    Parameters
    ----------
    model : tf.keras.Model
    dataset : tf.data.Dataset
    head : str
    pearson_per_target : np.ndarray
    out_dir : Path
    """
    logger = get_logger(
        name="VisualEvaluator",
        output_dir=out_dir,
        log_type="VisualEval",
        level="INFO",
    )

    logger.info("Collecting predictions for visualization...")
    y_true, y_pred = collect_predictions_for_visualization(
        model=model,
        dataset=dataset,
        head=head,
        max_batches=max_batches,
    )

    logger.info(f"Collected y_true/y_pred with shape {y_true.shape}")

    groups = select_representative_targets(pearson_per_target)

    for group, targets in groups.items():
        logger.info(f"Plotting {group} Pearson targets: {targets.tolist()}")
        plot_multiple_targets(
            y_true=y_true,
            y_pred=y_pred,
            target_indices=targets,
            sample_idx=sample_idx,
            out_dir=out_dir / group,
            prefix=group,
        )

    logger.info("Visual evaluation finished ✅")