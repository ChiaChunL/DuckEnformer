"""
Quick smoke-test demo: build DuckEnformer and run one forward pass on a
random one-hot DNA sequence. Verifies the environment is installed
correctly and the model builds/runs — does not require a trained
checkpoint or any downloaded data.
"""

import time

import numpy as np
import tensorflow as tf

from models.enformer import Enformer

SEQ_LENGTH = 196_608


def main():
    t0 = time.time()

    model = Enformer(
        channels=1536,
        num_heads=8,
        num_transformer_layers=11,
        pooling_type="max",
    )

    rng = np.random.default_rng(0)
    one_hot = np.zeros((1, SEQ_LENGTH, 4), dtype=np.float32)
    bases = rng.integers(0, 4, size=SEQ_LENGTH)
    one_hot[0, np.arange(SEQ_LENGTH), bases] = 1.0
    inputs = tf.convert_to_tensor(one_hot)

    outputs = model(inputs, is_training=False)
    duck_out = outputs["duck"]

    print(f"Output shape (duck head): {tuple(duck_out.shape)}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
