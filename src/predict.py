"""Apply a trained model + threshold to a candidate set to produce final matches."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from . import model as model_mod
from .features import FeatureTransformers, compute_features_parallel


def score_candidates(
    candidates: Dict[str, List[str]],
    s1_records: pd.DataFrame,
    cand_records: pd.DataFrame,
    clf,
    transformers: FeatureTransformers,
    n_jobs: int = -1,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    pairs = pd.DataFrame(
        [(s1, c) for s1, clist in candidates.items() for c in clist],
        columns=["s1_id", "cand_id"],
    )
    if pairs.empty:
        return pd.DataFrame(columns=["s1_id", "cand_id", "score"])
    feats = compute_features_parallel(
        pairs, s1_records, cand_records, transformers, n_jobs=n_jobs, chunk_size=chunk_size
    )
    feats["score"] = model_mod.predict_proba(clf, feats)
    return feats[["s1_id", "cand_id", "score"]]


def apply_threshold(
    scored: pd.DataFrame, threshold: float, s1_order: List[str]
) -> Dict[str, List[str]]:
    """Deterministic: matches sorted by descending score, then id, ties broken by id."""
    out: Dict[str, List[str]] = {s1: [] for s1 in s1_order}
    kept = scored[scored["score"] >= threshold].copy()
    kept = kept.sort_values(["s1_id", "score", "cand_id"], ascending=[True, False, True])
    for s1_id, group in kept.groupby("s1_id"):
        out[s1_id] = group["cand_id"].tolist()
    return out
