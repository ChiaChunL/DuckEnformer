"""
@File         :   DuckAlleleS.predictedSet.Construct.py
@Time         :   2026/01/16, 16:50
@Author       :   JiaJun Li
@Description  :   
"""

import argparse
from pathlib import Path
from utils.logger_util import get_logger

import pandas as pd
from tqdm import tqdm
from pyfaidx import Fasta

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

    # Paths & parameters
    GENOME_FASTA = args.genome_fa
    ASI_EFFECT_DIR = args.ASI_dir / "duck_epi_asi_effect"
    OUT_BED = args.Proj_dir / "data" / "processed" / "duckASI_l197k" / "Duck_ASI_variants_197k-ATAC-H3K27AC.bed"

    L = 196_608
    HALF = L // 2

    # Load genome
    logger.info(f"Loading genome FASTA: {GENOME_FASTA}")
    genome = Fasta(
        GENOME_FASTA,
        as_raw=True,
        sequence_always_upper=True,
    )

    logger.info(f"Loaded {len(genome.keys())} chromosomes from genome")

    # Collect all variants from ASI CSVs
    dfs = []

    csv_files = sorted(ASI_EFFECT_DIR.glob("*_ASI_effects.slim.csv"))

    # Select of interest
    interest_csvs = []

    for csv_file in tqdm(csv_files):
        name = csv_file.name

        is_atac = name.startswith("A-")
        is_h3k27ac = name.startswith("C-") and "-H3K27AC-" in name

        if is_atac or is_h3k27ac:
            interest_csvs.append(csv_file)

    csv_files = interest_csvs

    logger.info(f"Found {len(csv_files):,} ASI effect CSV files")

    # for test
    # csv_files = csv_files[:5]

    for csv in tqdm(csv_files, total=len(csv_files) ,desc="Reading ASI CSVs: ", unit=" ASI", dynamic_ncols=True):
        df = pd.read_csv(csv, usecols=["Chr", "Pos", "RefAllele", "AltAllele"])
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total variant rows (with duplicates): {len(df_all):,}")

    # Standardize chromosome name
    df_all["Chr"] = df_all["Chr"].astype(str)

    # Build variant key
    df_all["variant_key"] = df_all["Chr"] + ":" + df_all["Pos"].astype(str) + ":" + df_all["RefAllele"] + ":" + df_all["AltAllele"]

    # Deduplicate variants
    df_var = df_all.drop_duplicates("variant_key").copy()
    logger.info(f"Unique variants: {len(df_var):,}")

    primary_chrs = {str(i) for i in range(1, 41)} | {"Z"}
    before = len(df_var)
    df_var = df_var[df_var["Chr"].isin(primary_chrs)].copy()
    after = len(df_var)

    logger.info(
        f"Filtered to primary chromosomes (1–40, Z): "
        f"{before:,} → {after:,}"
    )

    # Ref allele validation against genome
    # --------------------------------------------------
    def ref_matches(row):
        try:
            base = genome[row["Chr"]][row["Pos"] - 1]
            return base == row["RefAllele"].upper()
        except KeyError:
            return False

    logger.info("Validating ref alleles against genome...")
    df_var["ref_match"] = df_var.apply(ref_matches, axis=1)
    ref_counts = df_var["ref_match"].value_counts()
    logger.info(f"Ref allele match summary:\n{ref_counts}")

    before = len(df_var)
    df_var = df_var[df_var["ref_match"]].copy()
    after = len(df_var)
    logger.info(f"Removed {before - after} variants due to ref mismatch")

    # Build centered windows (BED coordinates)
    df_var["center_0based"] = df_var["Pos"] - 1
    df_var["start"] = df_var["center_0based"] - HALF
    df_var["end"] = df_var["center_0based"] + HALF

    # Chromosome boundary check
    def valid_window(row):
        try:
            chrom_len = len(genome[row["Chr"]])
            return (row["start"] >= 0) and (row["end"] <= chrom_len)
        except KeyError:
            return False

    logger.info("Filtering windows outside chromosome boundaries...")
    before = len(df_var)
    df_var = df_var[df_var.apply(valid_window, axis=1)].copy()
    after = len(df_var)

    logger.info(f"Removed {before - after} variants due to boundary issues")

    # Sort BED entries (Chr, start, end)
    chr_order = {str(i): i for i in range(1, 41)}
    chr_order["Z"] = 41

    df_var["Chr_order"] = df_var["Chr"].map(chr_order)

    df_var = df_var.sort_values(by=["Chr_order", "start", "end"], ascending=True)
    df_var = df_var.drop(columns=["Chr_order"])

    # Write BED
    OUT_BED.parent.mkdir(parents=True, exist_ok=True)

    bed_cols = ["Chr", "start", "end", "variant_key"]
    df_var[bed_cols].to_csv(OUT_BED, sep="\t", index=False, header=False)

    logger.info(f"Final BED written to: {OUT_BED}")
    logger.info(f"Final variant count: {len(df_var):,}")
    logger.info("Duck ASI → Enformer BED preparation finished")



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

    parser.add_argument(
        "--genome_fa",
        type=Path,
        required=True,
        help="Path to reference genome FASTA (not distributed in this repo).",
    )

    args = parser.parse_args()
    main(args)
