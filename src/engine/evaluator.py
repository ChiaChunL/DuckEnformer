"""
@File         :   DuckEnformer/src/engine/evaluator.py
@Time         :   2025/12/28, 10:10
@Author       :   JiaJun Li
@Description  :   Evaluate Enformer model module.
"""
from typing import Optional
import tensorflow as tf
from tqdm import tqdm

# ======================
# Metrics
# ======================
class PearsonR(tf.keras.metrics.Metric):
    """
    Pearson correlation coefficient.
    reduce_axis=(0,1):
    (batch, genomic_bins) → per-target correlation.
    """

    def __init__(self, reduce_axis=(0, 1), name='PearsonR'):
        super().__init__(name=name)
        self.reduce_axis = reduce_axis
        self._initialized = False

    def _initialize(self, y_true):
        shape = [d for i, d in enumerate(y_true.shape)
                 if i not in self.reduce_axis]

        self.count = self.add_weight('count', shape=shape, initializer='zeros')
        self.sum_x = self.add_weight('sum_x', shape=shape, initializer='zeros')
        self.sum_y = self.add_weight('sum_y', shape=shape, initializer='zeros')
        self.sum_x2 = self.add_weight('sum_x2', shape=shape, initializer='zeros')
        self.sum_y2 = self.add_weight('sum_y2', shape=shape, initializer='zeros')
        self.sum_xy = self.add_weight('sum_xy', shape=shape, initializer='zeros')

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

        cov = (self.sum_xy
               - mean_x * self.sum_y
               - mean_y * self.sum_x
               + self.count * mean_x * mean_y)

        var_x = self.sum_x2 - self.count * tf.square(mean_x)
        var_y = self.sum_y2 - self.count * tf.square(mean_y)

        return cov / (tf.sqrt(var_x * var_y) + 1e-8)

    def reset_states(self):
        if self._initialized:
            for v in self.variables:
                v.assign(tf.zeros_like(v))


# ======================
# Evaluation
# ======================
def center_crop_target(target, target_len):
    total = tf.shape(target)[1]
    trim = (total - target_len) // 2
    return target[:, trim:trim + target_len, :]


@tf.function
def _predict_step(model, batch, head):
    preds = model(batch['sequence'], is_training=False)[head]
    targets = center_crop_target(batch['target'], tf.shape(preds)[1])
    return targets, preds


def evaluate_model(
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        head: str,
        max_steps: Optional[int] = None,
) -> tf.Tensor:
    """
    Returns
    -------
    pearson_per_target : tf.Tensor, shape (num_targets,)
    """
    metric = PearsonR(reduce_axis=(0, 1))
    metric.reset_states()

    if max_steps is not None:
        total = max_steps
    else:
        total = None  # tqdm 会退化为不定长

    for step, batch in tqdm(enumerate(dataset), total=total, desc="Evaluating", dynamic_ncols=True):
        if max_steps is not None and step >= max_steps:
            break
        targets, preds = _predict_step(model, batch, head)
        metric.update_state(targets, preds)

    return metric.result()
