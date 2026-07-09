"""
@File         :   DuckAlleleS.predict.effects.tidy.py
@Time         :   2026/01/18, 10:37
@Author       :   JiaJun Li
@Description  :   Parse Enformer delta npz files, extract center-bin effects,
                  and merge with ASI effect tables to build a long-format dataset.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.logger_util import get_logger

CENTER_BIN = 896 // 2  # 448


def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Run",
                      level="INFO")


def load_all_npz(npz_dirs, logger):
    """
    Load all npz files and extract:
    variant_key, delta_center (B, Track)
    """
    records = []

    for npz_dir in npz_dirs:
        logger.info(f"Scanning {npz_dir}")
        npz_files = sorted(npz_dir.glob("batch_*.npz"))
        logger.info(f"Found {len(npz_files):,} batches")

        for npz_file in tqdm(npz_files, desc=f"Reading {npz_dir.name}"):
            data = np.load(npz_file)

            variant_keys = data["variant_key"]  # (B,)
            delta = data["delta"]  # (B, 896, 240)
            delta_center = delta[:, CENTER_BIN, :]  # (B, 240)

            for i, vkey in enumerate(variant_keys):
                records.append(
                    {
                        "variant_key": vkey,
                        "delta_center": delta_center[i],  # ndarray (240,)
                    }
                )

    logger.info(f"Collected delta for {len(records):,} variants")
    return records


def expand_delta_records(records, track_meta, logger):
    """
    Expand each variant's (240,) delta into long-format rows
    """
    rows = []

    for rec in tqdm(records, desc="Expanding variant × track"):
        vkey = rec["variant_key"]
        deltas = rec["delta_center"]

        for track_idx, delta_val in enumerate(deltas):
            meta = track_meta.loc[track_idx]

            rows.append(
                {
                    "variant_key": vkey,
                    "TrackID": meta["TrackID"],
                    "TrackName": meta["TrackName"],
                    "Tissue": meta["Tissue"],
                    "Type": meta["Type"],
                    "delta": float(delta_val),
                }
            )

    df = pd.DataFrame(rows)
    logger.info(f"Expanded to {len(df):,} variant–track rows")
    return df


def main(args):
    logger = setup_logging(Path(__file__).stem, args.Proj_dir)

    logger.info("=" * 60)
    logger.info("Parsing Enformer delta predictions")
    logger.info(args)
    logger.info("=" * 60)

    # --------------------------------------------------
    # 1. Collect npz dirs
    # --------------------------------------------------
    npz_dirs = [
        args.infer_dir / "0",
        args.infer_dir / "1",
    ]

    # --------------------------------------------------
    # 2. Load track metadata (order MUST match model output)
    # --------------------------------------------------
    track_meta = pd.read_csv(
        args.track_meta,
        usecols=["Track", "RENAME", "Tissue", "Type"],
    ).rename(
        columns={
            "Track": "TrackID",
            "RENAME": "TrackName",
        }
    )

    track_meta = track_meta.reset_index(drop=True)
    logger.info(f"Loaded {len(track_meta)} track metadata rows")

    # --------------------------------------------------
    # 3. Load Enformer delta
    # --------------------------------------------------
    delta_records = load_all_npz(npz_dirs, logger)

    # --------------------------------------------------
    # 4. Expand to long-format
    # --------------------------------------------------
    df_delta = expand_delta_records(delta_records, track_meta, logger)

    # --------------------------------------------------
    # 5. Load ASI slim tables and merge
    # --------------------------------------------------
    asi_files = sorted(args.asi_effect_dir.glob("*_ASI_effects.slim.csv"))
    logger.info(f"Found {len(asi_files)} ASI effect files")

    asi_dfs = []

    for f in asi_files:
        df = pd.read_csv(f)

        df["variant_key"] = (
                df["Chr"].astype(str) + ":" +
                df["Pos"].astype(str) + ":" +
                df["RefAllele"] + ":" +
                df["AltAllele"]
        )

        asi_dfs.append(
            df[["variant_key", "TrackID", "ASI_log2_ratio", "ASI_alt_frac_centered", "ASI_direction", "pass_depth", "pass_fdr"]]
        )

    df_asi = pd.concat(asi_dfs, ignore_index=True)
    logger.info(f"ASI rows: {len(df_asi):,}")

    # --------------------------------------------------
    # 6. Merge delta × ASI
    # --------------------------------------------------
    df_final = df_delta.merge(
        df_asi,
        on=["variant_key", "TrackID"],
        how="inner",
    )

    logger.info(
        f"Final merged table: {df_final.shape[0]:,} rows"
    )

    # --------------------------------------------------
    # 7. Save
    # --------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"DuckEnformer_ASI_delta_long_{args.infer_dir.name}.csv"

    df_final.to_csv(out_file, index=False)
    logger.info(f"Saved final table to {out_file}")

    logger.info("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script description")

    parser.add_argument(
        "--Proj_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory",
    )

    parser.add_argument(
        "--infer_dir",
        type=Path,
        required=True,
        help="Inference result directory, e.g. results/inference/<run_id>",
    )

    parser.add_argument(
        "--asi_effect_dir",
        type=Path,
        required=True,
        help="ASI effect slim csv directory",
    )

    parser.add_argument(
        "--track_meta",
        type=Path,
        default=None,
        help="Duck_ASI_tracks_full.csv (defaults to <Proj_dir>/data/raw/Duck_ASI_tracks_full.csv)",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to <Proj_dir>/results/variant_effects)",
    )

    args = parser.parse_args()
    if args.track_meta is None:
        args.track_meta = args.Proj_dir / "data" / "raw" / "Duck_ASI_tracks_full.csv"
    if args.out_dir is None:
        args.out_dir = args.Proj_dir / "results" / "variant_effects"
    main(args)
