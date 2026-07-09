"""
@File         :   DuckEpigenomics.createTargets.py
@Time         :   2025/12/25, 10:36
@Author       :   JiaJun Li
@Description  :
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from utils.load_data import load_yaml
from utils.logger_util import get_logger
from datasets.mapping_target import map_encode_stat, create_desc_term


def setup_logging(script_name: str, proj_dir: Path):
    """Logger setup"""
    return get_logger(name=script_name,
                      output_dir=proj_dir / 'logs',
                      log_type="Run",
                      level="INFO")


def build_target(row, epi_type_format_files_path):
    epi_type = row.Type
    identifier = row.RENAME

    bw_file = (
        Path(epi_type_format_files_path[epi_type]["bw"])
        / f"{identifier}.bw"
    )

    clip, scale, sum_stat = map_encode_stat(epi_type, identifier)
    desc = create_desc_term(epi_type, row.Tissue, identifier, row.NAME)

    return {
        "index": row.Index,
        "genome": 0,
        "identifier": identifier,
        "file": str(bw_file),
        "clip": clip,
        "scale": scale,
        "sum_stat": sum_stat,
        "description": desc,
    }

def main(args):
    """
    Main entry point for both CLI and import usage.
    """
    logger = setup_logging(Path(__file__).stem, args.Proj_dir)

    logger.info("")
    logger.info(args)
    logger.info(f"{Path(__file__).stem} Start")
    logger.info("")


    epigenomics_data_config = args.Proj_dir / "configs" / "dataset" / "EpigenomicsData.yaml"
    epigenomics_config = load_yaml(epigenomics_data_config)

    epi_type_format_files_path = {
        k: v
        for k, v in epigenomics_config.items()
        if k not in {"detail"}
    }
    detailEpi = pd.read_csv(epigenomics_config['detail']['csv'], sep=",", header=0)
    logger.info(f"There are {detailEpi.shape[0]:,} epigenomics details")

    targets_list = []
    for row in tqdm(detailEpi.itertuples(index=True), total=len(detailEpi), desc="Processing targets"):
        targets_list.append(build_target(row, epi_type_format_files_path))

    targets_file = args.Proj_dir / "data" / "processed" / "enformerTgts" / "targets_duck.txt"
    targets_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(targets_list).to_csv(targets_file, index=False, header=True, sep="\t")
    logger.info(f"{len(targets_list)} targets created, saved on {targets_file}")


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

    args = parser.parse_args()
    main(args)
