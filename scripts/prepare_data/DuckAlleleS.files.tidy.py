"""
@File         :   DuckAlleleS.files.tidy.py
@Time         :   2026/01/16, 10:36
@Author       :   JiaJun Li
@Description  :   Load ASI mapping file, validate ASI result files,
                and generate a tidy ASI track metadata table for downstream modeling.
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import shutil

from utils.logger_util import get_logger


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

    # Load ASI mapping file
    mapping_file = args.ASI_dir / 'list_total_merge1'
    mappings = pd.read_csv(mapping_file, sep="\t", header=None, names=["prefix", "tissue_type"])
    logger.info(f"{len(mappings):,} mapping records loaded")
    logger.debug(mappings.head())

    # Construct ASI file paths
    count_dir = args.ASI_dir / "count_final"

    asi_terms = (
        mappings.assign(
            AS_file=lambda df: df["prefix"].apply(
                lambda p: count_dir / f"{p}_AS_significant_final.txt"
            )
        )
        .reset_index(drop=True)
    )

    # Load Duck epigenome metadata
    duck_epigenome_file = args.Proj_dir / "data" / "raw" / "DuckEpigenomics.csv"
    duck_epigenome_tracks = pd.read_csv(duck_epigenome_file, header=0, sep=",")

    logger.info(f"{len(duck_epigenome_tracks):,} epigenome tracks loaded")

    # Merge (FULL, no assay filtering)
    asi_term_tracks = pd.merge(duck_epigenome_tracks, asi_terms, left_on="NAME", right_on="prefix", how="left")

    # File existence check
    missing_mask = asi_term_tracks["AS_file"].notna() & \
                   (~asi_term_tracks["AS_file"].apply(Path.exists))

    if missing_mask.any():
        missing = asi_term_tracks.loc[missing_mask, ["NAME", "AS_file"]]
        logger.warning(
            f"{len(missing)} ASI files missing for epigenome tracks:\n"
            + "\n".join(
                f"{row.NAME}: {row.AS_file}"
                for row in missing.itertuples(index=False)
            )
        )

    asi_term_tracks = asi_term_tracks.loc[~missing_mask].reset_index(drop=True)
    logger.info(f"{len(asi_terms):,} ASI terms retained after file existence check")

    save_cols = ['Track', 'Type', 'NAME', 'RENAME', 'Tissue', 'AS_file', 'Desc']

    final_asi_details = asi_term_tracks.copy()
    logger.info(
        f"Final ASI table: "
        f"{final_asi_details['Track'].nunique()} tracks, "
        f"{final_asi_details['Tissue'].nunique()} tissues"
    )

    # Copy duck epigenome ASI files
    duck_epi_asi_file_save_dir = args.ASI_dir / "duck_epi_asi"
    duck_epi_asi_file_save_dir.mkdir(parents=True, exist_ok=True)

    # Construct new AS_file paths
    final_asi_details["New_AS_file"] = final_asi_details["RENAME"].apply(
        lambda name: duck_epi_asi_file_save_dir / f"{name}_AS_significant_final.txt"
    )

    logger.info(f"Copying {len(final_asi_details):,} ASI files to {duck_epi_asi_file_save_dir}")

    for row in tqdm(final_asi_details.itertuples(index=False), total=len(final_asi_details), desc="Copy ASI files",):
        src = Path(row.AS_file)
        dst = Path(row.New_AS_file)

        try:
            if not dst.exists():
                shutil.copy2(src, dst)
        except Exception as e:
            logger.error(f"Failed to copy {src} -> {dst}: {e}")
            raise

    # Replace AS_file column with new path
    final_asi_details = final_asi_details.drop(columns=["AS_file"]).rename(columns={"New_AS_file": "AS_file"})
    logger.info("ASI files copied and AS_file paths updated")

    out_file = args.Proj_dir / "data" / "raw" / "Duck_ASI_tracks_full.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    final_asi_details[save_cols].to_csv(out_file, index=False)
    logger.info(f"Full ASI track table saved to {out_file}")

    logger.info("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Duck Allele Specific imbalance sites tidy pipeline")

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
