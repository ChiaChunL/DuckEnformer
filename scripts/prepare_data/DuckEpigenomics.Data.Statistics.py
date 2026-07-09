"""
@File         :   DuckEpigenomics.Data.Statistics.py
@Time         :   2025/12/24, 10:14
@Author       :   JiaJun Li
@Description  :   Copy raw bam files into structured directories based on experiment type.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.utils.logger_util import get_logger


def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Run",
                      level="INFO")


def copy_bam_files(
        df: pd.DataFrame,
        epigen_dir: Path,
        logger,
        overwrite: bool = False,
        dry_run: bool = False,
):
    """
    Copy BAM files into structured directories based on experiment type.
    """
    type_to_subdir = {
        "ATAC-seq": epigen_dir / "atac" / "bam",
        "CUT&TAG": epigen_dir / "cuttag" / "bam",
    }

    # Ensure directories exist
    for d in type_to_subdir.values():
        d.mkdir(parents=True, exist_ok=True)

    copied, skipped, missing = 0, 0, 0

    for idx, row in df.iterrows():
        try:
            file_type = row["Type"]
            src = Path(row["Bam_file"]) / f"{row['NAME']}_deduplicate.bam"

            if file_type not in type_to_subdir:
                logger.warning(f"[SKIP] Unknown Type '{file_type}' (row {idx})")
                skipped += 1
                continue

            if not src.exists():
                logger.warning(f"[MISSING] {src}")
                missing += 1
                continue

            dst = type_to_subdir[file_type] / f"{row['RENAME']}.bam"

            if dst.exists() and not overwrite:
                logger.info(f"[EXIST] {dst} (skip)")
                skipped += 1
                continue

            if dry_run:
                logger.info(f"[DRY-RUN] {src} -> {dst}")
            else:
                shutil.copy2(src, dst)
                logger.info(f"[COPIED] {src.name} -> {dst}")

            copied += 1

        except Exception as e:
            logger.error(f"[ERROR] row {idx}: {e}", exc_info=True)

    logger.info("-" * 60)
    logger.info(f"Summary:")
    logger.info(f"  Copied : {copied}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Missing: {missing}")
    logger.info("-" * 60)


def main(args):
    """
    Main entry point for both CLI and import usage.
    """
    logger = setup_logging(Path(__file__).stem, args.Proj_dir)

    logger.info("")
    logger.info(args)
    logger.info(f"{Path(__file__).stem} Start")
    logger.info("")

    duckEpigenomics_filePath = args.Proj_dir / 'data' / 'raw' / 'DuckEpigenomics.csv'
    duckEpigenomics = pd.read_csv(duckEpigenomics_filePath, sep=",", header=0)
    logger.info(f"Loaded DuckEpigenomics table: {duckEpigenomics.shape[0]:,} rows")

    copy_bam_files(
        df=duckEpigenomics,
        epigen_dir=args.Epigen_dir,
        logger=logger,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    logger.info(f"{Path(__file__).stem} FINISHED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script description")

    parser.add_argument(
        "--Proj_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory",
    )

    parser.add_argument(
        "--Epigen_dir",
        type=Path,
        required=True,
        help="Epigenomes directory",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing BAM files",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print operations without copying files",
    )
    args = parser.parse_args()
    main(args)
