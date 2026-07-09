"""
@File         :   DuckEpigenomics.bam2bw.py
@Time         :   2025/12/24, 11:31
@Author       :   JiaJun Li
@Description  :   
"""
import argparse
from pathlib import Path
from src.utils.logger_util import get_logger
import subprocess

def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Run",
                      level="INFO")


def ensure_bam_index(bam: Path, logger, threads: int = 8, dry_run: bool = False):
    """
    Ensure BAM index (.bai) exists; create it if missing.
    """
    bai = bam.with_suffix(".bam.bai")

    if bai.exists():
        return

    logger.info(f"🔧 Indexing BAM: {bam.name}")
    cmd = [
        "samtools", "index",
        "-@", str(threads),
        str(bam)
    ]

    if dry_run:
        logger.info(f"[dry-run] {' '.join(cmd)}")
        return

    subprocess.run(cmd, check=True)


def main(args):
    """
    Main entry point for both CLI and import usage.
    """
    logger = setup_logging(Path(__file__).stem, args.Proj_dir)

    logger.info("")
    logger.info(args)
    logger.info(f"{Path(__file__).stem} Start")
    logger.info("")

    bam_subdir = {
        "ATAC-seq": args.Epigen_dir / "atac" / "bam",
        "CUT&TAG": args.Epigen_dir / "cuttag" / "bam",
    }
    bw_subdir = {
        "ATAC-seq": args.Epigen_dir / "atac" / "bw",
        "CUT&TAG": args.Epigen_dir / "cuttag" / "bw",
    }

    GENOME_SIZE = 1211992756
    THREADS = 16
    NORMALIZATION = "RPGC"

    epi_types = ["ATAC-seq", "CUT&TAG"]
    for epi_type in epi_types:
        bam_epi_dir = bam_subdir[epi_type]
        bw_epi_dir = bw_subdir[epi_type]
        bw_epi_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"BAM |{bam_epi_dir}| → BW |{bw_epi_dir}|")

        for bam in bam_epi_dir.glob("*.bam"):
            sample = bam.stem
            bw = bw_epi_dir / f"{sample}.bw"
            log = bw_epi_dir / f"{sample}.bamCoverage.log"

            if bw.exists() and not args.overwrite:
                logger.info(f"⏭️  Skip existing {bw.name} (use --overwrite to force)")
                continue

            if bw.exists() and args.overwrite:
                logger.info(f"🔁 Overwrite enabled: {bw.name}")

            ensure_bam_index(bam, logger, THREADS, dry_run=args.dry_run)

            bin_size = 10 if epi_type == "ATAC-seq" else 25

            cmd = [
                "bamCoverage",
                "-b", str(bam),
                "-o", str(bw),
                "--binSize", str(bin_size),
                "--normalizeUsing", NORMALIZATION,
                "--numberOfProcessors", str(THREADS),
                "--effectiveGenomeSize", str(GENOME_SIZE),
                "--ignoreForNormalization", "chrM",
            ]

            if epi_type == "ATAC-seq":
                cmd += ["--extendReads", "--centerReads"]

            logger.info(f"▶ [{epi_type}] Processing {sample}")

            logger.info(f"CMD: {' '.join(cmd)}")

            if args.dry_run:
                logger.info("[dry-run] bamCoverage not executed")
                continue

            with open(log, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=f, check=True)


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
        help="Overwrite existing bigWig files",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands without executing",
    )

    args = parser.parse_args()
    main(args)
