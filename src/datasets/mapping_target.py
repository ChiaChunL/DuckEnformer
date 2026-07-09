# ./src/datasets/mapping_target.py
from typing import Tuple


def map_encode_stat(encode_type: str, identifier: str) -> Tuple[int, int, str]:
    """
    Map epigenomic assay type + mark identifier to
    (window_size, stride, pooling_method) for signal binning.

    window_size : genomic bin size (bp)
    stride      : pooling stride for down-sampling
    pool_method : how to aggregate signal within each window

    Design principle:
    - sharp / sparse signals → small window, sum
    - broad / continuous signals → larger window
    - noisy signals → mean + stride
    """
    encode_type = encode_type.upper()
    identifier = identifier.upper()

    # CAGE: extremely sharp TSS signal, preserve total counts
    if encode_type == "CAGE":
        return 384, 1, "sum"

    # DNase-seq: narrow accessibility peaks, count cleavage events
    if encode_type == "DNASE":
        return 64, 1, "sum"

    # ATAC-seq: noisy accessibility, average with down-sampling
    if encode_type == "ATAC-SEQ":
        return 32, 4, "mean"

    # CUT&Tag / ChIP-seq: histone modification signals
    if encode_type == "CUT&TAG" or encode_type == "CHIP-SEQ":

        # Broad repressive / elongation marks
        if any(m in identifier for m in ["H3K27ME3", "H3K36ME3"]):
            return 64, 1, "sum"

        # Sharp promoter / enhancer marks
        if any(m in identifier for m in ["H3K4ME3", "H3K27AC", "H3K4ME1"]):
            return 32, 2, "mean"

        # CUT&TAG fallback, default histone mark handling
        return 32, 1, "mean"

    # Global fallback for unknown assays
    return 32, 1, "mean"


def create_desc_term(encode_type: str, tissue: str, identifier: str, raw_file: str) -> str:
    identifier_upper = identifier.upper()
    mark_encode = ""

    for mark in ["H3K4ME3", "H3K27AC", "H3K4ME1", "H3K27ME3"]:
        if mark in identifier_upper:
            mark_encode = mark
            break

    desc_term = f"{encode_type}:{tissue} | {mark_encode} {identifier} from {raw_file}"
    return desc_term
