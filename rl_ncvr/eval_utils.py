"""
Evaluation utilities for retrieval on NCVR without extra persistence.

Design:
- Use the same serialization as training (markers chosen via tokenizer probe).
- For a given box, for each source S, query S against candidates from other sources.
- Memory-safe retrieval: stream candidate embeddings in shards, maintain top-K
  per query across shards. No FAISS dependency.

Metrics:
- Recall@K for K in {1, 10, 50}
- MRR@10
- Threshold precision/recall/F1 using top-1 neighbor cosine similarities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from rl_ncvr.box_splitting import get_box, DEFAULT_DATA_DIR
from rl_ncvr.pair_generation import _try_detect_marker_words, _serialize_row


@dataclass
class EvalParams:
    box_id: int
    model_name_or_path: str = "nreimers/MiniLM-L6-H384-uncased"
    recid_column: str = "recid"
    entity_columns: Sequence[str] = ("givenname", "surname", "postcode", "suburb")
    data_dir: Optional[str] = None
    device: Optional[str] = None  # 'cuda' or 'cpu'; auto-detect if None
    batch_size: int = 2048
    candidate_chunk_size: int = 200_000
    ks: Sequence[int] = (1, 10, 50)


def _serialize_parts_for_box(
    params: EvalParams,
) -> Dict[str, pd.DataFrame]:
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
            result[src] = df
            continue
        cols_to_keep = [params.recid_column, *params.entity_columns]
        missing = [c for c in params.entity_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns in {src}: {missing}")
        view = df[cols_to_keep].copy()
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
    return result


def _load_model_and_tokenizer(model_name_or_path: str, device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def _encode_texts(texts: List[str], model, tokenizer, device: str, batch_size: int) -> np.ndarray:
    embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        token_embeddings = outputs[0]
        input_mask_expanded = enc['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embs.append(pooled.detach().cpu().numpy())
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 384), dtype=np.float32)


def _topk_across_shards(
    query_embs: np.ndarray,
    candidate_iter: Iterable[Tuple[np.ndarray, np.ndarray]],  # yields (cand_embs, cand_recids)
    ks: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Maintain top-K hits per query across multiple candidate shards.
    Returns dicts: top_scores[k] -> (Q, k), top_recids[k] -> (Q, k)
    """
    Q = query_embs.shape[0]
    max_k = max(ks)
    # Initialize with very low scores and empty recids
    top_scores = {k: np.full((Q, k), -np.inf, dtype=np.float32) for k in ks}
    top_recids = {k: np.empty((Q, k), dtype=object) for k in ks}

    # We'll maintain for max_k, then slice for each K
    cur_scores = np.full((Q, max_k), -np.inf, dtype=np.float32)
    cur_recids = np.empty((Q, max_k), dtype=object)

    for cand_embs, cand_recids in candidate_iter:
        # Compute similarities (cosine = dot because unit-normalized)
        sims = np.matmul(query_embs, cand_embs.T)  # (Q, C)
        # For each query, merge candidates into existing top list
        # Compute indices of top max_k along axis 1
        # Combine current top and new sims
        combined_scores = np.concatenate([cur_scores, sims], axis=1)  # (Q, max_k + C)
        combined_recids = np.concatenate([cur_recids, np.broadcast_to(cand_recids, (Q, cand_recids.shape[0]))], axis=1)
        # Argsort descending
        idx = np.argpartition(combined_scores, -max_k, axis=1)[:, -max_k:]
        # Gather top candidates
        row_idx = np.arange(Q)[:, None]
        top_s = combined_scores[row_idx, idx]
        top_r = combined_recids[row_idx, idx]
        # Now sort each row descending fully
        order = np.argsort(-top_s, axis=1)
        cur_scores = np.take_along_axis(top_s, order, axis=1)
        cur_recids = np.take_along_axis(top_r, order, axis=1)

    for k in ks:
        top_scores[k] = cur_scores[:, :k]
        top_recids[k] = cur_recids[:, :k]
    return top_scores, top_recids


def _compute_metrics(
    query_recids: np.ndarray,
    top_recids_at_k: Dict[int, np.ndarray],
    top_scores_at_k: Dict[int, np.ndarray],
    ks: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    Q = len(query_recids)
    # Recall@K
    for k in ks:
        hits = np.any(top_recids_at_k[k] == query_recids[:, None], axis=1)
        metrics[f"recall@{k}"] = float(np.mean(hits))
    # MRR@10
    ranks = np.full(Q, np.inf)
    top10 = top_recids_at_k.get(10)
    if top10 is not None:
        for i in range(Q):
            match = np.where(top10[i] == query_recids[i])[0]
            if match.size > 0:
                ranks[i] = match[0] + 1
        finite = ranks[np.isfinite(ranks)]
        metrics["mrr@10"] = float(np.mean(1.0 / finite)) if finite.size > 0 else 0.0
    else:
        metrics["mrr@10"] = float('nan')
    # PR/F1 using top-1
    top1_scores = top_scores_at_k[1][:, 0]
    top1_recids = top_recids_at_k[1][:, 0]
    labels = (top1_recids == query_recids)
    # Threshold sweep
    thresholds = np.unique(top1_scores)
    best_f1 = 0.0
    best_thr = 0.0
    best_prec = 0.0
    best_rec = 0.0
    for thr in thresholds:
        preds = top1_scores >= thr
        tp = np.sum(preds & labels)
        fp = np.sum(preds & ~labels)
        fn = np.sum(~preds & labels)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr, best_prec, best_rec = f1, thr, prec, rec
    metrics.update({
        "clf_best_f1": float(best_f1),
        "clf_best_threshold": float(best_thr),
        "clf_precision_at_best_f1": float(best_prec),
        "clf_recall_at_best_f1": float(best_rec),
    })
    return metrics


def evaluate_box(params: EvalParams) -> Dict[str, float]:
    """
    Evaluate retrieval metrics for a given box. Returns aggregated metrics.
    """
    parts = _serialize_parts_for_box(params)
    model, tokenizer, device = _load_model_and_tokenizer(params.model_name_or_path, params.device)

    # Aggregate metrics across sources
    all_metrics: List[Dict[str, float]] = []
    sources = list(parts.keys())
    for query_src in sources:
        # Query set
        q_df = parts[query_src]
        if q_df.empty:
            continue
        q_recids = q_df[params.recid_column].astype(str).values
        q_texts = q_df["text"].tolist()
        q_embs = _encode_texts(q_texts, model, tokenizer, device, params.batch_size)

        # Candidate pool = all other sources
        cand_dfs = [parts[s] for s in sources if s != query_src and not parts[s].empty]
        if not cand_dfs:
            continue
        cand_all = pd.concat(cand_dfs, ignore_index=True)
        cand_recids_all = cand_all[params.recid_column].astype(str).values
        cand_texts_all = cand_all["text"].tolist()

        # Iterate candidate shards
        def shard_iter() -> Iterable[Tuple[np.ndarray, np.ndarray]]:
            for i in range(0, len(cand_texts_all), params.candidate_chunk_size):
                texts_chunk = cand_texts_all[i:i + params.candidate_chunk_size]
                recids_chunk = cand_recids_all[i:i + params.candidate_chunk_size]
                embs_chunk = _encode_texts(texts_chunk, model, tokenizer, device, params.batch_size)
                yield embs_chunk.astype(np.float32, copy=False), recids_chunk

        top_scores_at_k, top_recids_at_k = _topk_across_shards(q_embs, shard_iter(), params.ks)
        metrics = _compute_metrics(q_recids, top_recids_at_k, top_scores_at_k, params.ks)
        all_metrics.append(metrics)

    # Aggregate by mean
    if not all_metrics:
        return {"recall@1": 0.0, "recall@10": 0.0, "recall@50": 0.0, "mrr@10": 0.0, "clf_best_f1": 0.0}
    keys = all_metrics[0].keys()
    agg = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    return agg


__all__ = [
    "EvalParams",
    "evaluate_box",
]


