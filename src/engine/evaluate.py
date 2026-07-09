"""
@File         :   evaluate.py
@Time         :   2025/12/19, 10:55
@Author       :   JiaJun Li
@Description  :
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf

from src.datasets.dataset import get_dataset

from src.utils.logger_util import get_logger
from src.models import enformer

# ======================
# Metrics
# ======================
class PearsonR(tf.keras.metrics.Metric):
    def __init__(self, reduce_axis=(0, 1), name='PearsonR'):
        super().__init__(name=name)
        self.reduce_axis = reduce_axis
        self._initialized = False

    def _initialize(self, y_true):
        # remaining shape after reduction
        shape = [
            d for i, d in enumerate(y_true.shape)
            if i not in self.reduce_axis
        ]

        self.count = self.add_weight(
            name='count', shape=shape, initializer='zeros'
        )
        self.sum_x = self.add_weight(
            name='sum_x', shape=shape, initializer='zeros'
        )
        self.sum_y = self.add_weight(
            name='sum_y', shape=shape, initializer='zeros'
        )
        self.sum_x2 = self.add_weight(
            name='sum_x2', shape=shape, initializer='zeros'
        )
        self.sum_y2 = self.add_weight(
            name='sum_y2', shape=shape, initializer='zeros'
        )
        self.sum_xy = self.add_weight(
            name='sum_xy', shape=shape, initializer='zeros'
        )

        self._initialized = True

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if not self._initialized:
            self._initialize(y_true)

        self.sum_x.assign_add(tf.reduce_sum(y_true, axis=self.reduce_axis))
        self.sum_y.assign_add(tf.reduce_sum(y_pred, axis=self.reduce_axis))
        self.sum_x2.assign_add(tf.reduce_sum(tf.square(y_true), axis=self.reduce_axis))
        self.sum_y2.assign_add(tf.reduce_sum(tf.square(y_pred), axis=self.reduce_axis))
        self.sum_xy.assign_add(tf.reduce_sum(y_true * y_pred, axis=self.reduce_axis))
        self.count.assign_add(tf.reduce_sum(tf.ones_like(y_true), axis=self.reduce_axis))

    def result(self):
        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count

        cov = (
            self.sum_xy
            - mean_x * self.sum_y
            - mean_y * self.sum_x
            + self.count * mean_x * mean_y
        )

        var_x = self.sum_x2 - self.count * tf.square(mean_x)
        var_y = self.sum_y2 - self.count * tf.square(mean_y)

        return cov / (tf.sqrt(var_x * var_y) + 1e-8)

    def reset_states(self):
        if self._initialized:
            for v in self.variables:
                v.assign(tf.zeros_like(v))

# ======================
# Evaluation function
# ======================

def center_crop_target(target, target_len):
    total = tf.shape(target)[1]
    trim = (total - target_len) // 2
    return target[:, trim:trim+target_len, :]


def evaluate_model(model, dataset, head, max_steps=None):
    metric = PearsonR(reduce_axis=(0, 1))

    for i, batch in tqdm(enumerate(dataset), desc="Evaluating", total=max_steps):
        if max_steps and i >= max_steps:
            break
        preds = model(batch['sequence'], is_training=False)[head]
        targets = center_crop_target(batch['target'], tf.shape(preds)[1])
        metric.update_state(targets, preds)

    return metric.result()



def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Run",
                      level="INFO")

# ======================
# CLI
# ======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_dir', type=Path, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    logger = get_logger(
        name="evaluate",
        output_dir=args.proj_dir / "logs",
        log_type="Eval",
        level="INFO"
    )

    dataset = (
        get_dataset('duck', args.split)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = enformer.Enformer(
        channels=1536 // 4,
        num_heads=8,
        num_transformer_layers=11,
        pooling_type='max'
    )

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.ckpt).expect_partial()

    pearson = evaluate_model(model, dataset, head='duck', max_steps=args.max_steps)
    result = {
        "mean": float(pearson.numpy().mean()),
        "per_target": pearson.numpy().tolist()
    }

    logger.info(f"PearsonR mean = {result['mean']:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)