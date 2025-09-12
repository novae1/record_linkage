"""
Generate positive pairs from a box (default: box 0) of the NCVR dataset
without persisting intermediate dataframes. Produces a .json.gz file where
each line is a JSON list [text_a, text_b] for training.

Constraints implemented per user requirements:
- Only serialize: givenname, surname, postcode, suburb (exclude recid)
- Join across 5 sources by recid and create all cross-source pairs
- Drop entities that yield no distinct pairs (implicitly, by yielding none)
- Serialize rows to a single string using markers derived from common vocab
- Drop identical pairs (keep only pairs where serialized strings differ)
 - Drop identical pairs (keep only pairs where serialized strings differ)
 - Globally shuffle output lines to de-cluster pairs from the same entity

Output example line:
    ["[FIELD] givenname [VALUE] John [FIELD] surname [VALUE] Smith ...",
     "[FIELD] givenname [VALUE] Jon [FIELD] surname [VALUE] Smyth ..."]

CLI usage (run from project root):
    python -m rl_ncvr.pair_generation \
        --box_id 0 \
        --output "rl_ncvr/outputs/box0_positive_pairs.json.gz" \
        --data_dir "data/north_carolina_voters"

Training script config (example data_config.json):
    [
      {"name": "box0_positive_pairs.json.gz", "weight": 1}
    ]

Launch training (example, run from project root):
    python rl_ncvr/train_script.py \
        --model nreimers/MiniLM-L6-H384-uncased \
        --steps 2000 \
        --batch_size 64 \
        --data_folder "rl_ncvr/outputs" \
        "rl_ncvr/outputs/data_config.json" \
        "outputs/your_experiment_folder"
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from box_splitting import get_box, DEFAULT_DATA_DIR


def _try_detect_marker_words(model_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Try to pick marker words that are likely in-vocab as single tokens.
    Preference order: (col, val) -> (field, value) -> (column, value) -> (key, value)
    If transformers/tokenizer is unavailable, default to (field, value).
    Returns a tuple of uppercase marker words, e.g., ("FIELD", "VALUE").
    """
    candidates = [("col", "val"), ("field", "value"), ("column", "value"), ("key", "value")]
    try:
        from transformers import AutoTokenizer
        name = model_name or "nreimers/MiniLM-L6-H384-uncased"
        tok = AutoTokenizer.from_pretrained(name)
        for a, b in candidates:
            ta = tok.tokenize(a)
            tb = tok.tokenize(b)
            if len(ta) == 1 and len(tb) == 1 and ta[0] != tok.unk_token and tb[0] != tok.unk_token:
                return a.upper(), b.upper()
        return "FIELD", "VALUE"
    except Exception:
        return "FIELD", "VALUE"


def _serialize_row(
    row: pd.Series,
    *,
    columns: Sequence[str],
    field_marker: str,
    value_marker: str,
    null_literal: str = "null",
) -> str:
    """
    Serialize a row to a single standardized string using markers.
    Output format: "[FIELD] col1 [VALUE] val1 [FIELD] col2 [VALUE] val2 ..."
    """
    parts: List[str] = []
    for col in columns:
        val = row.get(col, None)
        if pd.isna(val):
            sval = null_literal
        else:
            sval = str(val).strip()
            sval = sval if sval != "" else null_literal
        parts.append(f"[{field_marker}] {col} [{value_marker}] {sval}")
    return " ".join(parts)


def _yield_pairs_for_entity(
    group_df: pd.DataFrame,
    *,
    columns: Sequence[str],
    field_marker: str,
    value_marker: str,
) -> Iterable[Tuple[str, str]]:
    """
    Yield all distinct serialized pairs for a single entity (recid group).
    Drops identical pairs (string equality).
    """
    if len(group_df) <= 1:
        return []
    serialized: List[str] = [
        _serialize_row(row, columns=columns, field_marker=field_marker, value_marker=value_marker)
        for _, row in group_df.iterrows()
    ]
    emitted = 0
    for i, j in combinations(range(len(serialized)), 2):
        a, b = serialized[i], serialized[j]
        if a != b:
            emitted += 1
            yield a, b
    if emitted == 0:
        return []


def generate_positive_pairs_for_box(
    *,
    box_id: int = 0,
    data_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    recid_column: str = "recid",
    entity_columns: Sequence[str] = ("givenname", "surname", "postcode", "suburb"),
    chunksize_hint: int = 100_000,
) -> str:
    """
    Build positive pairs for the specified box and write them to a .json.gz file.

    Behavior:
    - For each entity (`recid` group), serialize selected columns and emit all
      pairwise combinations where the two serialized strings differ.
    - Accumulate all pairs across entities in-memory, then globally shuffle the
      list before writing a line-delimited JSON gzip file. This reduces the
      chance that multiple pairs from the same entity appear near one another.

    Returns the absolute path to the written output file.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    if output_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"box{box_id}_positive_pairs.json.gz")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    field_marker, value_marker = _try_detect_marker_words(tokenizer_model)

    # Load 5 dataframes for the box in-memory (filtered already by recid->box)
    parts: Dict[str, pd.DataFrame] = get_box(
        box_id,
        data_dir=data_dir,
        recid_column=recid_column,
        chunksize=chunksize_hint,
    )

    # Keep only recid + desired columns and add a source identifier
    trimmed_parts: List[pd.DataFrame] = []
    for src_name, df in parts.items():
        if df.empty:
            continue
        cols_to_keep = [c for c in [recid_column, *entity_columns] if c in df.columns]
        missing = [c for c in entity_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns in {src_name}: {missing}")
        trimmed = df[cols_to_keep].copy()
        trimmed_parts.append(trimmed)

    if not trimmed_parts:
        # Nothing to write; create an empty gzip with zero lines
        with gzip.open(output_path, "wt") as f_out:
            pass
        return os.path.abspath(output_path)

    all_rows = pd.concat(trimmed_parts, ignore_index=True)

    # Group by recid, collect pairs, then shuffle globally before writing
    total_pairs = 0
    total_entities = 0
    total_entities_with_pairs = 0
    buffered_pairs: List[Tuple[str, str]] = []
    for _recid_value, group in all_rows.groupby(recid_column, sort=False):
        total_entities += 1
        emitted_here = 0
        for a, b in _yield_pairs_for_entity(
            group,
            columns=entity_columns,
            field_marker=field_marker,
            value_marker=value_marker,
        ):
            buffered_pairs.append((a, b))
            total_pairs += 1
            emitted_here += 1
        if emitted_here > 0:
            total_entities_with_pairs += 1

    # Shuffle globally to minimize locality of pairs from the same entity
    random.shuffle(buffered_pairs)

    with gzip.open(output_path, "wt") as f_out:
        for a, b in buffered_pairs:
            f_out.write(json.dumps([a, b]) + "\n")

    print(
        f"Wrote {total_pairs:,} pairs from {total_entities:,} entities "
        f"({total_entities_with_pairs:,} with pairs) to {output_path}"
    )
    return os.path.abspath(output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate positive pairs for a box and write .json.gz")
    p.add_argument("--box_id", type=int, default=0)
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write .json.gz (defaults to rl_ncvr/outputs/box{box}_positive_pairs.json.gz)",
    )
    p.add_argument("--tokenizer_model", type=str, default=None, help="Optional tokenizer to probe for marker words")
    return p


def main():
    args = _build_arg_parser().parse_args()
    generate_positive_pairs_for_box(
        box_id=args.box_id,
        data_dir=args.data_dir,
        output_path=args.output,
        tokenizer_model=args.tokenizer_model,
    )


if __name__ == "__main__":
    main()


