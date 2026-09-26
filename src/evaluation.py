"""Labeling, splitting, and the macro-F0.5 evaluation used by the pipeline."""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def build_pairwise_labels(
    candidates: Dict[str, List[str]], ground_truth: Dict[str, List[str]]
) -> pd.DataFrame:
    """Turn (candidates, ground_truth) into a labeled (s1_id, cand_id, label) table.

    Every candidate pair gets a label: 1 if it's a true match per ground truth,
    0 otherwise (this naturally yields hard negatives -- high-similarity
    candidates that block together with the true match but aren't it).
    """
    rows = []
    for s1_id, cand_ids in candidates.items():
        true_set = set(ground_truth.get(s1_id, []))
        for cid in cand_ids:
            rows.append((s1_id, cid, 1 if cid in true_set else 0))
    return pd.DataFrame(rows, columns=["s1_id", "cand_id", "label"])


def candidate_recall(
    candidates: Dict[str, List[str]], ground_truth: Dict[str, List[str]]
) -> Tuple[float, int, int]:
    """Fraction of true-match ids that are present in the candidate set."""
    total_true = 0
    found = 0
    for s1_id, true_ids in ground_truth.items():
        if not true_ids:
            continue
        cand_set = set(candidates.get(s1_id, []))
        for t in true_ids:
            total_true += 1
            if t in cand_set:
                found += 1
    recall = found / total_true if total_true else 1.0
    return recall, found, total_true


def group_split(
    s1_ids: List[str], val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    """Split by SOURCE1 ENTITY (not by individual pair) so no entity's pairs
    leak across train/val."""
    rng = np.random.RandomState(seed)
    ids = list(s1_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:n_val])
    train_ids = [i for i in ids if i not in val_ids]
    val_ids = [i for i in ids if i in val_ids]
    return train_ids, val_ids


def _entity_f_beta(pred: Set[str], true: Set[str], beta: float = 0.5) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    inter = len(pred & true)
    precision = inter / len(pred)
    recall = inter / len(true)
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f_beta(
    predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]], s1_ids: List[str], beta: float = 0.5
) -> Dict[str, float]:
    """Macro-averaged F-beta (default beta=0.5) over source1 entities, plus
    micro precision/recall for diagnostics."""
    scores = []
    tp = fp = fn = 0
    for s1_id in s1_ids:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())
        scores.append(_entity_f_beta(pred, true, beta))
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)
    macro = float(np.mean(scores)) if scores else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f_beta = (
        (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
        if (precision > 0 or recall > 0)
        else 0.0
    )
    return {
        "macro_f0.5": macro,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f0.5": f_beta,
        "n_entities": len(s1_ids),
    }


def singleton_performance(
    predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]], s1_ids: List[str]
) -> Dict[str, float]:
    """Restrict macro F0.5 to entities with exactly one true match."""
    singleton_ids = [i for i in s1_ids if len(ground_truth.get(i, set())) == 1]
    if not singleton_ids:
        return {"singleton_macro_f0.5": None, "n_singletons": 0}
    metrics = macro_f_beta(predictions, ground_truth, singleton_ids, beta=0.5)
    return {"singleton_macro_f0.5": metrics["macro_f0.5"], "n_singletons": len(singleton_ids)}


def predictions_from_scores(
    scored: pd.DataFrame, threshold: float
) -> Dict[str, Set[str]]:
    """scored must have columns s1_id, cand_id, score."""
    kept = scored[scored["score"] >= threshold]
    out: Dict[str, Set[str]] = {}
    for s1_id, group in kept.groupby("s1_id"):
        out[s1_id] = set(group["cand_id"].tolist())
    return out


def tune_threshold(
    scored: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    s1_ids: List[str],
    thresholds: List[float],
) -> Tuple[float, pd.DataFrame]:
    """Try each threshold, return the one with the best macro F0.5 plus a
    full results table for reporting."""
    rows = []
    best_t, best_f05 = thresholds[0], -1.0
    for t in thresholds:
        preds = predictions_from_scores(scored, t)
        metrics = macro_f_beta(preds, ground_truth, s1_ids, beta=0.5)
        rows.append({"threshold": t, **metrics})
        if metrics["macro_f0.5"] > best_f05:
            best_f05 = metrics["macro_f0.5"]
            best_t = t
    return best_t, pd.DataFrame(rows)
