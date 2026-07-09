"""
@File         :   inference.py
@Time         :   2026/01/17, 14:55
@Author       :   JiaJun Li
@Description  :
"""

import tensorflow as tf
import numpy as np

@tf.function
def infer(model, input):
    return model(input, is_training=False)

def flush_batch(model, batch_ref, batch_alt, batch_keys, batch_id, out_dir):
    if not batch_ref:
        return
    x_ref = tf.convert_to_tensor(np.stack(batch_ref), dtype=tf.float32)  # (B,L,4)
    x_alt = tf.convert_to_tensor(np.stack(batch_alt), dtype=tf.float32)

    pred_ref = infer(model, x_ref)['duck']
    pred_alt = infer(model, x_alt)['duck']

    # # concat for single forward
    # x = tf.concat([x_ref, x_alt], axis=0)
    # pred = infer(model, x)["duck"]
    # pred_ref, pred_alt = tf.split(pred, 2, axis=0)

    delta = pred_alt - pred_ref

    np.savez(
        out_dir / f"batch_{batch_id:06d}.npz",
        batch_id=batch_id,
        variant_key=np.array(batch_keys),
        # pred_ref=pred_ref.numpy(),
        # pred_alt=pred_alt.numpy(),
        delta=delta.numpy(),
    )

    batch_ref.clear()
    batch_alt.clear()
    batch_keys.clear()