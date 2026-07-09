"""
@File         :   dataset.py
@Time         :   2025/12/19, 10:54
@Author       :   JiaJun Li
@Description  :
"""

import os
import json
import functools
from pathlib import Path

import tensorflow as tf
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
config_path = os.environ.get("DUCKENFORMER_CONFIG", str(_REPO_ROOT / "configs" / "duck.yaml"))
with open(config_path) as f:
    cfg = yaml.safe_load(f)


def organism_path(organism):
    return os.path.join(cfg['data']['dataset_root'], organism)


def get_metadata(organism):
    path = os.path.join(organism_path(organism), 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def tfrecord_files(organism, subset):
    return sorted(
        tf.io.gfile.glob(
            os.path.join(
                organism_path(organism),
                'tfrecords',
                f'{subset}-*.tfr'
            )
        ),
        key=lambda x: int(x.split('-')[-1].split('.')[0])
    )


def deserialize(serialized_example, metadata):
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)

    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(
        target,
        (metadata['target_length'], metadata['num_targets'])
    )
    target = tf.cast(target, tf.float32)

    return {'sequence': sequence, 'target': target}


def get_dataset(organism, subset, num_threads=8):
    metadata = get_metadata(organism)
    dataset = tf.data.TFRecordDataset(
        tfrecord_files(organism, subset),
        compression_type='ZLIB',
        num_parallel_reads=num_threads
    )
    dataset = dataset.map(
        functools.partial(deserialize, metadata=metadata),
        num_parallel_calls=num_threads
    )
    return dataset
