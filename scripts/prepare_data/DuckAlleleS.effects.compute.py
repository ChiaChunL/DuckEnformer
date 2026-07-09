"""
@File         :   DuckAlleleS.effects.compute.py
@Time         :   2026/01/16, 16:12
@Author       :   JiaJun Li
@Description  :   
"""

import argparse
from pathlib import Path
from utils.logger_util import get_logger
import pandas as pd
import numpy as np
from  tqdm import tqdm


def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Run",
                      level="INFO")


def main(args):
    logger = setup_logging(Path(__file__).stem, args.Proj_dir)

    logger.info("")
    logger.info(args)
    logger.info(f"{Path(__file__).stem} Start")
    logger.info("")

    # Used files Path
    duck_epi_asi_file = args.Proj_dir / 'data' / 'raw' / 'Duck_ASI_tracks_full.csv'
    out_dir = args.ASI_dir / "duck_epi_asi_effect"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load track metadata
    tracks = pd.read_csv(duck_epi_asi_file, header=0, sep=",")
    logger.info(f"{tracks.shape[0]} tracks loaded")

    # Iterate over tracks
    for row in tqdm(tracks.itertuples(index=False), total=len(tracks), desc="Processing ASI tracks: "):

        asi_file = Path(row.AS_file)
        if not asi_file.exists():
            continue

        # ---- load ASI txt ----
        df = pd.read_csv(asi_file, sep="\t")
        df = df.rename(
            columns={
                "contig": "Chr",
                "position": "Pos",
                "refAllele": "RefAllele",
                "altAllele": "AltAllele",
                "refCount": "RefCount",
                "altCount": "AltCount",
                "totalCount": "totalCount",
            }
        )

        # ---- add track metadata ----
        df["TrackID"] = row.Track
        df["TrackName"] = row.RENAME
        df["Tissue"] = row.Tissue
        df["Type"] = row.Type
        df["Pos_1based"] = df["Pos"]
        df["Pos_0based"] = df["Pos"] - 1
        df["variant_key"] = df["Chr"].astype(str).str.cat(
            [
                df["Pos_1based"].astype(str),
                df["RefAllele"],
                df["AltAllele"],
            ],
            sep=":"
        )
        df["track_variant_key"] = df["TrackID"].astype(str).str.cat(
            df["variant_key"], sep="|"
        )
        # ---- compute ASI effects ----
        df["ASI_log2_ratio"] = np.log2(
            (df["AltCount"] + 0.5) / (df["RefCount"] + 0.5)
        )

        df["ASI_alt_frac_centered"] = (
                df["AltCount"] / df["totalCount"] - 0.5
        )

        df["ASI_direction"] = np.sign(df["ASI_log2_ratio"])

        # ---- QC flags ----
        df["pass_depth"] = df["totalCount"] >= 50
        df["pass_fdr"] = df["fdr"] < 0.05

        # ---- save ----
        SLIM_COLS = [
            "TrackID", "TrackName", "Tissue", "Type",
            "Chr", "Pos", "RefAllele", "AltAllele",
            "RefCount", "AltCount", "totalCount", "fdr",
            "ASI_log2_ratio", "ASI_alt_frac_centered", "ASI_direction",
            "pass_depth", "pass_fdr",
        ]
        df_slim = df[SLIM_COLS].copy()
        df_slim.to_csv(out_dir / f"{row.RENAME}_ASI_effects.slim.csv", index=False)
        df.to_csv(out_dir / f"{row.RENAME}_ASI_effects.raw.csv", index=False)

    logger.info(f"All duck epigenome tracks with asi effects computed, saved on {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script description")

    parser.add_argument(
        "--Proj_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory",
    )

    parser.add_argument(
        "--ASI_dir",
        type=Path,
        required=True,
        help="Allele-specific imbalance files directory",
    )

    args = parser.parse_args()
    main(args)
