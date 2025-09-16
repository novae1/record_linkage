"""
Record-level recall@K evaluation utilities for NCVR blocking.

Design:
- Reuse training serialization (markers chosen via tokenizer probe).
- For a given box, evaluate each source as queries against all other sources.
- Memory-safe retrieval: stream candidate embeddings in shards and maintain
  top-K per query across shards. No FAISS dependency.

Metric:
- Record-level recall@K (easy, hard). For a query with m true matches in the
  candidate pool (same recid) and n matches in top-K, recall@K = n / m.
  Easy counts all matches; Hard excludes candidates identical to the query text.
  Records with m == 0 are excluded from the average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import hashlib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from box_splitting import get_box, DEFAULT_DATA_DIR
from pair_generation import _try_detect_marker_words, _serialize_row


@dataclass
class EvalParams:
    """
    Minimal configuration for record-level recall@K evaluation.

    Attributes
    ----------
    box_id : int
        The NCVR box id to evaluate.
    model_name_or_path : str
        HF model for encoding texts.
    recid_column : str
        Column name identifying entities.
    entity_columns : Sequence[str]
        Columns to serialize into text.
    data_dir : Optional[str]
        Directory containing NCVR CSVs (defaults to packaged DEFAULT_DATA_DIR).
    device : Optional[str]
        'cuda' or 'cpu'. If None, auto-detect.
    batch_size : int
        Tokenization/encoding batch size.
    candidate_chunk_size : int
        Size of candidate shards to encode per pass.
    verbose : bool
        Print progress logs.
    max_length : int
        Tokenizer max length.
    sources_to_eval : Optional[Sequence[str]]
        Restrict evaluation to specific sources; None evaluates all.
    """

    box_id: int
    model_name_or_path: str = "nreimers/MiniLM-L6-H384-uncased"
    recid_column: str = "recid"
    entity_columns: Sequence[str] = ("givenname", "surname", "postcode", "suburb")
    data_dir: Optional[str] = None
    device: Optional[str] = None  # 'cuda' or 'cpu'; auto-detect if None
    batch_size: int = 2048
    candidate_chunk_size: int = 200_000
    verbose: bool = True
    max_length: int = 128
    sources_to_eval: Optional[Sequence[str]] = None
    # Optional micro-shard controls (filter entities by recid hash)
    shard_modulus: Optional[int] = None
    shard_remainder: Optional[int] = None


def _serialize_parts_for_box(params: EvalParams) -> Dict[str, pd.DataFrame]:
    """
    Load and serialize all sources for the given box using training serialization.

    Returns a dict mapping source name to a DataFrame with columns
    [recid_column, "text"]. Empty sources are preserved.
    """
    if params.verbose:
        print(
            f"[eval] Loading and serializing box={params.box_id} from {params.data_dir or DEFAULT_DATA_DIR}..."
        )
    parts = get_box(
        params.box_id,
        data_dir=params.data_dir or DEFAULT_DATA_DIR,
        recid_column=params.recid_column,
        chunksize=100_000,
    )
    field_marker, value_marker = _try_detect_marker_words(params.model_name_or_path)
    result: Dict[str, pd.DataFrame] = {}
    for src, df in parts.items():
        if df.empty:
            if params.verbose:
                print(f"[eval] Source {src}: 0 rows")
            result[src] = df
            continue
        cols_to_keep = [params.recid_column, *params.entity_columns]
        missing = [c for c in params.entity_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns in {src}: {missing}")
        view = df[cols_to_keep].copy()
        # Optional recid-based micro-sharding (keeps entity structure across sources)
        if params.shard_modulus is not None:
            remainder = 0 if params.shard_remainder is None else params.shard_remainder
            if params.verbose:
                print(
                    f"[eval]  -> Applying shard filter: recid % {params.shard_modulus} == {remainder}"
                )
            recids = view[params.recid_column].astype(str).values
            def sha256_uint64(s: str) -> int:
                d = hashlib.sha256(s.encode('utf-8')).digest()
                return int.from_bytes(d[:8], byteorder='big', signed=False)
            h = np.fromiter((sha256_uint64(s) for s in recids), dtype=np.uint64, count=len(recids))
            mask = (h % np.uint64(params.shard_modulus)) == np.uint64(remainder)
            view = view.loc[mask].reset_index(drop=True)
        view["text"] = view.apply(
            lambda row: _serialize_row(
                row,
                columns=params.entity_columns,
                field_marker=field_marker,
                value_marker=value_marker,
            ),
            axis=1,
        )
        result[src] = view[[params.recid_column, "text"]]
        if params.verbose:
            print(f"[eval] Source {src}: {len(result[src]):,} rows serialized")
    return result


def _load_model_and_tokenizer(
    model_name_or_path: str,
    device: Optional[str] = None,
) -> Tuple[AutoModel, AutoTokenizer, str]:
    """
    Load a HF model/tokenizer and put the model on the requested or auto device.

    Returns a tuple (model, tokenizer, resolved_device).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if device:
        print(f"[eval] Loaded model '{model_name_or_path}' on device={device}")
    return model, tokenizer, device


@torch.no_grad()
def _encode_texts(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """
    Encode a list of strings into L2-normalized mean-pooled embeddings.
    """
    embs: List[np.ndarray] = []
    # Coerce to a plain Python list of strings for the tokenizer
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    else:
        try:
            texts = list(texts)
        except TypeError:
            texts = [texts]
    texts = [str(t) for t in texts]
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        token_embeddings = outputs[0]
        input_mask_expanded = (
            enc["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embs.append(pooled.detach().cpu().numpy())
        # Lightweight progress every ~50 batches
        if batch_size > 0:
            bidx = i // batch_size
            if bidx % 50 == 0 and bidx > 0:
                print(
                    f"[eval] Encoded {min(i + batch_size, total):,}/{total:,} texts..."
                )
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 384), dtype=np.float32)


def _topk_across_shards(
    query_embs: np.ndarray,
    candidate_iter: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ks: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Maintain top-K (by cosine similarity) per query across multiple candidate shards.

    Returns dicts: top_scores[k] -> (Q, k), top_recids[k] -> (Q, k), top_texts[k] -> (Q, k).
    """
    Q = query_embs.shape[0]
    max_k = max(ks)
    # Initialize with very low scores and empty recids
    top_scores = {k: np.full((Q, k), -np.inf, dtype=np.float32) for k in ks}
    top_recids = {k: np.empty((Q, k), dtype=object) for k in ks}
    top_texts = {k: np.empty((Q, k), dtype=object) for k in ks}

    # We'll maintain for max_k, then slice for each K
    cur_scores = np.full((Q, max_k), -np.inf, dtype=np.float32)
    cur_recids = np.empty((Q, max_k), dtype=object)
    cur_texts = np.empty((Q, max_k), dtype=object)

    for cand_embs, cand_recids, cand_texts in candidate_iter:
        # Compute similarities (cosine = dot because unit-normalized)
        sims = np.matmul(query_embs, cand_embs.T)  # (Q, C)
        # Combine current top and new sims
        combined_scores = np.concatenate([cur_scores, sims], axis=1)  # (Q, max_k + C)
        combined_recids = np.concatenate(
            [cur_recids, np.broadcast_to(cand_recids, (Q, cand_recids.shape[0]))], axis=1
        )
        combined_texts = np.concatenate(
            [cur_texts, np.broadcast_to(cand_texts, (Q, cand_texts.shape[0]))], axis=1
        )
        # Select top max_k indices per row (partial sort)
        idx = np.argpartition(combined_scores, -max_k, axis=1)[:, -max_k:]
        row_idx = np.arange(Q)[:, None]
        top_s = combined_scores[row_idx, idx]
        top_r = combined_recids[row_idx, idx]
        top_t = combined_texts[row_idx, idx]
        # Order them descending
        order = np.argsort(-top_s, axis=1)
        cur_scores = np.take_along_axis(top_s, order, axis=1)
        cur_recids = np.take_along_axis(top_r, order, axis=1)
        cur_texts = np.take_along_axis(top_t, order, axis=1)

    for k in ks:
        top_scores[k] = cur_scores[:, :k]
        top_recids[k] = cur_recids[:, :k]
        top_texts[k] = cur_texts[:, :k]
    return top_scores, top_recids, top_texts


def evaluate_box_record_recall(
    params: EvalParams,
    ks: Sequence[int],
) -> Dict[str, float]:
    """
    Compute record-level recall@K (easy, hard) for blocking quality.

    For each query, let m be the total number of matching records in the
    candidate pool (same recid). For a K, let n be the number of matches among
    the top-K retrieved candidates. The record-level recall@K is n / m.

    Variants:
    - Easy: counts all matching candidates (same recid)
    - Hard: excludes candidates whose serialized text equals the query text

    Queries with m == 0 are excluded from the average.

    Returns a dict with keys like record_recall@{K}_easy and
    record_recall@{K}_hard (global averages across sources).
    """
    # Load serialized parts and model
    parts = _serialize_parts_for_box(params)
    model, tokenizer, device = _load_model_and_tokenizer(
        params.model_name_or_path, params.device
    )

    # Prepare accumulators across all sources
    per_k_scores_easy: Dict[int, List[float]] = {int(k): [] for k in ks}
    per_k_scores_hard: Dict[int, List[float]] = {int(k): [] for k in ks}

    sources_all = list(parts.keys())
    sources = list(params.sources_to_eval) if params.sources_to_eval else sources_all

    for query_src in sources:
        q_df = parts[query_src]
        if q_df.empty:
            continue

        q_recids = q_df[params.recid_column].astype(str).values
        q_texts = q_df["text"].astype(str).values

        if params.verbose:
            print(
                f"[eval-rr] Encoding queries from {query_src}: {len(q_texts):,} rows..."
            )
        q_embs = _encode_texts(
            list(q_texts), model, tokenizer, device, params.batch_size, params.max_length
        )

        # Candidate pool = all other sources
        cand_dfs = [
            parts[s] for s in sources if s != query_src and not parts[s].empty
        ]
        if not cand_dfs:
            if params.verbose:
                print(f"[eval-rr] Skip candidates for {query_src}: no other sources")
            continue
        cand_all = pd.concat(cand_dfs, ignore_index=True)
        cand_recids_all = cand_all[params.recid_column].astype(str).values
        cand_texts_all = cand_all["text"].astype(str).values

        # Build denominator helpers
        recid_total: Dict[str, int] = {}
        recid_text_counts: Dict[str, Dict[str, int]] = {}
        for r, t in zip(cand_recids_all.tolist(), cand_texts_all.tolist()):
            recid_total[r] = recid_total.get(r, 0) + 1
            d = recid_text_counts.get(r)
            if d is None:
                d = {}
                recid_text_counts[r] = d
            d[t] = d.get(t, 0) + 1

        if params.verbose:
            total_c = len(cand_texts_all)
            shards = math.ceil(total_c / params.candidate_chunk_size)
            print(
                f"[eval-rr] Candidates for {query_src}: {total_c:,} rows across {shards} shard(s) of up to {params.candidate_chunk_size:,}."
            )

        def shard_iter() -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            total = len(cand_texts_all)
            shards = math.ceil(total / params.candidate_chunk_size)
            for si, i in enumerate(
                range(0, total, params.candidate_chunk_size), start=1
            ):
                texts_chunk = cand_texts_all[i : i + params.candidate_chunk_size]
                recids_chunk = cand_recids_all[i : i + params.candidate_chunk_size]
                if params.verbose:
                    print(
                        f"[eval-rr]  -> Encoding candidate shard {si}/{shards} (size {len(texts_chunk):,})..."
                    )
                embs_chunk = _encode_texts(
                    texts_chunk.tolist(),
                    model,
                    tokenizer,
                    device,
                    params.batch_size,
                    params.max_length,
                )
                if params.verbose:
                    print(
                        f"[eval-rr]  -> Encoded shard {si}/{shards}, merging top-K..."
                    )
                yield embs_chunk.astype(np.float32, copy=False), recids_chunk, texts_chunk

        top_scores_at_k, top_recids_at_k, top_texts_at_k = _topk_across_shards(
            q_embs, shard_iter(), ks
        )

        # Compute denominators for all queries (easy and hard)
        m_easy = np.array([recid_total.get(r, 0) for r in q_recids], dtype=np.int32)
        # For hard, exclude candidate rows identical to the query's text
        m_hard = np.array(
            [recid_total.get(r, 0) - (recid_text_counts.get(r, {}).get(qt, 0)) for r, qt in zip(q_recids, q_texts)],
            dtype=np.int32,
        )

        # Valid masks: queries with at least one true match in candidates
        mask_easy = m_easy > 0
        mask_hard = m_hard > 0

        q_recids_col = q_recids[:, None]
        q_texts_col = q_texts[:, None]

        for k in ks:
            trec = top_recids_at_k[int(k)]  # (Q, k)
            ttxt = top_texts_at_k[int(k)]  # (Q, k)
            # Easy numerator: count matches by recid among top-k
            match_easy = trec == q_recids_col
            n_easy = match_easy.sum(axis=1).astype(np.int32)
            # Hard numerator: matches by recid but exclude identical texts
            non_identical = ttxt != q_texts_col
            match_hard = match_easy & non_identical
            n_hard = match_hard.sum(axis=1).astype(np.int32)

            # Record-level recall for eligible queries
            rec_easy = (
                n_easy[mask_easy] / m_easy[mask_easy].astype(np.float32)
                if mask_easy.any()
                else np.array([], dtype=np.float32)
            )
            rec_hard = (
                n_hard[mask_hard] / m_hard[mask_hard].astype(np.float32)
                if mask_hard.any()
                else np.array([], dtype=np.float32)
            )

            per_k_scores_easy[int(k)].extend(rec_easy.tolist())
            per_k_scores_hard[int(k)].extend(rec_hard.tolist())

    # Aggregate globally across all sources
    results: Dict[str, float] = {}
    for k in ks:
        arr_e = np.array(per_k_scores_easy[int(k)], dtype=np.float32)
        arr_h = np.array(per_k_scores_hard[int(k)], dtype=np.float32)
        results[f"record_recall@{int(k)}_easy"] = float(arr_e.mean()) if arr_e.size > 0 else 0.0
        results[f"record_recall@{int(k)}_hard"] = float(arr_h.mean()) if arr_h.size > 0 else 0.0

    if params.verbose:
        print(f"[eval-rr] Aggregated record-level recall: {results}")
    return results


__all__ = [
    "EvalParams",
    "evaluate_box_record_recall",
]


