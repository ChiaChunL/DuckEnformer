"""
@File         :   inference.py
@Time         :   2025/12/20
@Author       :   JiaJun Li
@Description  :   Run Enformer inference from local checkpoint (best/) and local config (duck.yaml).
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Dict

import yaml
import numpy as np
import tensorflow as tf

from src.utils.logger_util import get_logger
from src.models.enformer import Enformer


# -----------------------------
# Utilities
# -----------------------------
def setup_logging(script_name: str, proj_dir: Path):
    return get_logger(
        name=script_name,
        output_dir=proj_dir / "logs",
        log_type="Run",
        level="INFO",
    )


def set_visible_gpus(gpu: Optional[str], logger=None):
    """
    Set CUDA_VISIBLE_DEVICES before TF initializes GPUs.
    gpu:
      - None: do nothing
      - "0" / "1,2,3": set CUDA_VISIBLE_DEVICES
      - "cpu": force CPU
    """
    if gpu is None:
        return

    if gpu.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if logger:
            logger.info("Using CPU only (CUDA_VISIBLE_DEVICES='').")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        if logger:
            logger.info(f"CUDA_VISIBLE_DEVICES set to '{gpu}'.")


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_ckpt_config(ckpt_dir: Path) -> Path:
    """
    Prefer ckpt_dir/duck.yaml. If not present, fall back to ckpt_dir/../config.yaml.
    """
    p1 = ckpt_dir / "duck.yaml"
    if p1.exists():
        return p1

    p2 = ckpt_dir.parent / "config.yaml"
    if p2.exists():
        return p2

    raise FileNotFoundError(
        f"Cannot find config in:\n- {p1}\n- {p2}\n"
        f"Please ensure you copied the training config into the checkpoint folder."
    )


def build_model_from_cfg(cfg: Dict) -> Enformer:
    """
    Strictly build model from cfg to match checkpoint shapes.
    Expected keys:
      cfg['model']['channels']
      cfg['model']['num_heads']
      cfg['model']['num_transformer_layers']
      cfg['model']['pooling_type']
    """
    m = cfg.get("model", {})
    required = ["channels", "num_heads", "num_transformer_layers", "pooling_type"]
    missing = [k for k in required if k not in m]
    if missing:
        raise KeyError(f"Config missing model keys: {missing}")

    return Enformer(
        channels=int(m["channels"]),
        num_heads=int(m["num_heads"]),
        num_transformer_layers=int(m["num_transformer_layers"]),
        pooling_type=str(m["pooling_type"]),
    )


def restore_model(model: tf.Module, ckpt_dir: Path, logger=None):
    ckpt = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(str(ckpt_dir))
    if latest is None:
        raise RuntimeError(f"No checkpoint found under: {ckpt_dir}")

    if logger:
        logger.info(f"Restoring from: {latest}")

    # expect_partial() is fine if you didn't track optimizer/global_step here.
    ckpt.restore(latest).expect_partial()

    if logger:
        logger.info("Model loaded successfully ✅")


def read_fasta_first_sequence(fasta_path: Path) -> str:
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    seq = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq.append(line)
    s = "".join(seq).upper()
    if not s:
        raise ValueError(f"No sequence found in FASTA: {fasta_path}")
    return s


def one_hot_encode_dna(seq: str, length: int) -> np.ndarray:
    """
    Produce one-hot [length, 4] in order A C G T, N -> all zeros.
    If seq shorter than length: center-pad with N (zeros).
    If seq longer than length: center-crop.
    """
    seq = seq.upper().replace("U", "T")
    L = len(seq)

    if L < length:
        # center pad
        pad_total = length - L
        left = pad_total // 2
        right = pad_total - left
        seq = ("N" * left) + seq + ("N" * right)
    elif L > length:
        # center crop
        start = (L - length) // 2
        seq = seq[start: start + length]

    assert len(seq) == length

    mapping = {
        "A": 0, "C": 1, "G": 2, "T": 3,
    }
    arr = np.zeros((length, 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = mapping.get(ch, None)
        if j is not None:
            arr[i, j] = 1.0
    return arr


def make_dummy_input(batch_size: int, seq_length: int) -> tf.Tensor:
    """
    Dummy input for quick sanity check: [B, seq_length, 4] zeros.
    用途：
      - 验证 checkpoint 能不能 restore
      - 验证推理流程是否通
      - 不依赖真实 DNA 序列
    """
    return tf.zeros((batch_size, seq_length, 4), dtype=tf.float32)


@tf.function
def predict(model: Enformer, inputs: tf.Tensor, head: str = "duck") -> tf.Tensor:
    """
    Run model forward.
    Returns tensor [B, 896, num_targets] for duck head.
    """
    outputs = model(inputs, is_training=False)
    if head not in outputs:
        raise KeyError(f"Head '{head}' not found. Available heads: {list(outputs.keys())}")
    return outputs[head]


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Local Enformer inference from checkpoint(best/).")

    p.add_argument(
        "--proj_dir",
        type=Path,
        default=Path.cwd(),
        help="Project root (for logs, optional). Defaults to current working directory.",
    )

    p.add_argument(
        "--ckpt_dir",
        type=Path,
        required=True,
        help="Checkpoint directory, e.g. checkpoints/duck_enformer/<run_id>/best",
    )

    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU ids for CUDA_VISIBLE_DEVICES, e.g. '0' or '1,2,3'. Use 'cpu' to force CPU. Default: unset (use all visible GPUs).",
    )

    p.add_argument(
        "--head",
        type=str,
        default="duck",
        help="Which head to use (default: duck).",
    )

    # input options
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--dna_seq", type=str, default=None, help="DNA sequence string (A/C/G/T/N).")
    g.add_argument("--fasta", type=Path, default=None, help="FASTA file (use first sequence).")
    g.add_argument("--dummy", action="store_true", help="Use dummy zero input.")

    p.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    p.add_argument(
        "--seq_length",
        type=int,
        default=None,
        help="Override input seq_length. If not set, will try cfg['data']['seq_length'] else fall back to 196608.",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npz path. Default: <ckpt_dir>/preds_<head>.npz",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # set gpu visibility ASAP (before TF lists devices)
    # NOTE: logger not ready yet, so pass None first; we will log afterwards too.
    set_visible_gpus(args.gpu, logger=None)

    script_name = Path(__file__).stem
    logger = setup_logging(script_name, args.proj_dir)

    logger.info("===================================")
    logger.info(f"Args: {args}")
    logger.info("===================================")

    # Load config next to checkpoint
    cfg_path = find_ckpt_config(args.ckpt_dir)
    logger.info(f"Loading config: {cfg_path}")
    cfg = load_yaml(cfg_path)

    # Decide sequence length
    seq_length = args.seq_length
    if seq_length is None:
        seq_length = int(cfg.get("data", {}).get("seq_length", 196_608))
    logger.info(f"Using seq_length = {seq_length}")

    # Build & restore model
    logger.info("Building model from config...")
    model = build_model_from_cfg(cfg)

    logger.info("Loading model from checkpoint...")
    restore_model(model, args.ckpt_dir, logger=logger)

    # Prepare inputs
    if args.dummy or (args.dna_seq is None and args.fasta is None):
        logger.info("Using dummy input (all zeros).")
        inputs = make_dummy_input(args.batch_size, seq_length)

    else:
        if args.fasta is not None:
            dna = read_fasta_first_sequence(args.fasta)
            logger.info(f"Loaded FASTA: {args.fasta} (len={len(dna)})")
        else:
            dna = args.dna_seq
            logger.info(f"Using dna_seq from CLI (len={len(dna)})")

        onehot = one_hot_encode_dna(dna, length=seq_length)  # [L,4]
        onehot = np.expand_dims(onehot, axis=0)  # [1,L,4]
        if args.batch_size != 1:
            onehot = np.repeat(onehot, repeats=args.batch_size, axis=0)
        inputs = tf.convert_to_tensor(onehot, dtype=tf.float32)

    logger.info(f"Input tensor shape: {inputs.shape}, dtype: {inputs.dtype}")

    # Run inference
    logger.info("Running inference...")
    y = predict(model, inputs, head=args.head)  # [B, 896, C]
    y_np = y.numpy()
    logger.info(f"Pred shape: {y_np.shape}, dtype: {y_np.dtype}")

    # Save
    out_path = args.out
    if out_path is None:
        out_path = args.ckpt_dir / f"preds_{args.head}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, preds=y_np)
    logger.info(f"Saved predictions to: {out_path}")

    logger.info("Done ✅")


if __name__ == "__main__":
    main()
