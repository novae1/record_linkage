"""
Utilities for deterministic, on-demand entity-based splitting of the
North Carolina Voters (NCVR) dataset into 10 disjoint 'boxes' without
writing any new data files.

Core design:
- Each entity is identified by its recId. We map each normalized recId
  to a box via a stable cryptographic hash modulo the number of boxes.
- get_box(box_id) streams and filters each source CSV to return 5 DataFrames
  containing only rows whose recId maps to the requested box.
- get_boxes(box_ids) does the same for multiple boxes at once.

No data persistence is performed; everything is computed on-the-fly.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


# Default absolute path to the NCVR CSV files in this workspace
DEFAULT_DATA_DIR = \
    "/home/nicolas/Documents/record_linkage/data/north_carolina_voters"


def normalize_recid(recid: Union[str, int, float, None]) -> Optional[str]:
    """
    Normalize recId to a lowercase, stripped string. Returns None if missing.
    """
    if recid is None:
        return None
    # Convert numerics safely to string without scientific notation
    if isinstance(recid, float):
        if pd.isna(recid):
            return None
        # Represent as integer-like if possible, else as plain string
        if recid.is_integer():
            recid = str(int(recid))
        else:
            recid = format(recid, 'f').rstrip('0').rstrip('.')
    else:
        recid = str(recid)
    normalized = recid.strip().lower()
    return normalized if normalized != "" else None


def compute_box_id(
    recid: Union[str, int, float, None],
    *,
    num_boxes: int = 10,
    salt: str = "ncvr_v1",
) -> Optional[int]:
    """
    Deterministically map a recId to a box in [0, num_boxes-1] using SHA-256.

    A small salt string is included to lock the mapping schema.
    Returns None for missing/invalid recIds.
    """
    normalized = normalize_recid(recid)
    if normalized is None:
        return None
    digest = hashlib.sha256(f"{salt}|{normalized}".encode("utf-8")).digest()
    # Use first 8 bytes for a fast stable integer
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer % num_boxes


def _discover_source_files(
    data_dir: Optional[str] = None,
) -> List[str]:
    """
    Discover NCVR CSV source files in the given directory.

    Returns a list of absolute file paths sorted by filename for stability.
    """
    base = data_dir or DEFAULT_DATA_DIR
    if not os.path.isabs(base):
        base = os.path.abspath(base)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"NCVR data directory not found: {base}. Please set data_dir correctly."
        )
    files = [
        os.path.join(base, f)
        for f in os.listdir(base)
        if f.lower().endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {base}. Expected 5 source CSVs."
        )
    files.sort()
    return files


def _filter_file_for_boxes(
    file_path: str,
    box_ids: Sequence[int],
    *,
    num_boxes: int = 10,
    recid_column: str = "recid",
    chunksize: int = 100_000,
    usecols: Optional[Sequence[str]] = None,
    dtype_overrides: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Stream filter a single CSV by keeping only rows whose recId maps to one of
    the provided box_ids. Returns a concatenated DataFrame for the file.
    """
    if dtype_overrides is None:
        dtype_overrides = {}
    # Ensure recId is read as string to avoid type issues
    if recid_column not in dtype_overrides:
        dtype_overrides[recid_column] = "string"

    keep_ids = set(box_ids)
    collected: List[pd.DataFrame] = []

    # Read in chunks to keep memory in check
    reader = pd.read_csv(
        file_path,
        dtype=dtype_overrides,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=True,
    )

    for chunk in reader:
        # Normalize recId and compute box
        recids = chunk[recid_column].astype("string")
        boxes = recids.map(lambda x: compute_box_id(x, num_boxes=num_boxes))
        mask = boxes.isin(keep_ids)
        if mask.any():
            collected.append(chunk.loc[mask])

    if not collected:
        # Return an empty DataFrame with the proper columns
        # We need to infer columns; re-read a tiny sample
        sample = pd.read_csv(file_path, nrows=0)
        return sample.iloc[:0]

    return pd.concat(collected, ignore_index=True)


def get_box(
    box_id: int,
    *,
    num_boxes: int = 10,
    data_dir: Optional[str] = None,
    recid_column: str = "recid",
    chunksize: int = 100_000,
    usecols: Optional[Sequence[str]] = None,
    dtype_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Return a dict mapping each source filename to a DataFrame that contains only
    the rows belonging to the requested box_id. No files are written.
    """
    if box_id < 0 or box_id >= num_boxes:
        raise ValueError(f"box_id must be in [0, {num_boxes-1}]")

    files = _discover_source_files(data_dir)
    result: Dict[str, pd.DataFrame] = {}
    for fp in files:
        df = _filter_file_for_boxes(
            fp,
            [box_id],
            num_boxes=num_boxes,
            recid_column=recid_column,
            chunksize=chunksize,
            usecols=usecols,
            dtype_overrides=dtype_overrides,
        )
        result[os.path.basename(fp)] = df
    return result


def get_boxes(
    box_ids: Sequence[int],
    *,
    num_boxes: int = 10,
    data_dir: Optional[str] = None,
    recid_column: str = "recid",
    chunksize: int = 100_000,
    usecols: Optional[Sequence[str]] = None,
    dtype_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Return a dict mapping each source filename to a DataFrame that contains rows
    belonging to any of the provided box_ids. No files are written.
    """
    if not box_ids:
        raise ValueError("box_ids must be non-empty")
    invalid = [b for b in box_ids if b < 0 or b >= num_boxes]
    if invalid:
        raise ValueError(
            f"All box_ids must be in [0, {num_boxes-1}]. Invalid: {invalid}"
        )

    files = _discover_source_files(data_dir)
    result: Dict[str, pd.DataFrame] = {}
    for fp in files:
        df = _filter_file_for_boxes(
            fp,
            box_ids,
            num_boxes=num_boxes,
            recid_column=recid_column,
            chunksize=chunksize,
            usecols=usecols,
            dtype_overrides=dtype_overrides,
        )
        result[os.path.basename(fp)] = df
    return result


def quick_validate_counts(
    *,
    num_boxes: int = 10,
    data_dir: Optional[str] = None,
    recid_column: str = "recid",
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Lightweight validation that estimates balance by counting rows per box per
    source. This does not require loading full DataFrames into memory.

    Returns a tidy DataFrame with columns: [source, box_id, row_count]
    """
    files = _discover_source_files(data_dir)
    records: List[Tuple[str, int, int]] = []  # (source, box_id, row_count)

    for fp in files:
        counts = {b: 0 for b in range(num_boxes)}
        reader = pd.read_csv(
            fp,
            dtype={recid_column: "string"},
            usecols=[recid_column],
            chunksize=chunksize,
            low_memory=True,
        )
        for chunk in reader:
            boxes = chunk[recid_column].astype("string").map(
                lambda x: compute_box_id(x, num_boxes=num_boxes)
            )
            vc = boxes.value_counts(dropna=True)
            for b, n in vc.items():
                if b is not None:
                    counts[int(b)] += int(n)
        src = os.path.basename(fp)
        for b in range(num_boxes):
            records.append((src, b, counts[b]))

    df = pd.DataFrame(records, columns=["source", "box_id", "row_count"])
    return df


__all__ = [
    "normalize_recid",
    "compute_box_id",
    "get_box",
    "get_boxes",
    "quick_validate_counts",
]


