"""Candidate generation (blocking) for source1 x {source2, source3}.

Design notes (see README for details): the datasets here are large
(millions of rows per source), so blocking is done with vectorized
pandas merges/joins rather than any O(n*m) python loop, and is
partitioned by country first (matches were empirically verified to
never cross countries in the ground truth -- this is a blocking
optimization, not a hard filter: every country present in the data is
still processed, nothing is excluded). This keeps every rule below
close to linear in the partition size.
"""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from . import normalization as norm

MAX_BLOCK_SIZE = 400          # cap for high-precision keys (exact name, prefix,
                               # first-token, postal) -- these rarely explode
HIGH_FREQUNCY_BLOCK_CAP = 25  # much tighter cap for noisy/generic keys (shared
                               # numeric tokens, address-word tokens) -- common
                               # words like city/state names or "road"/"floor"
                               # otherwise blow up to MAX_BLOCK_SIZE^2 pairs per key
NAME_NGRAM_RANGE = (2, 4)
TFIDF_BUCKET_MAX = 3000        # cap on the size of a sub-bucket fed into the
                               # dense-ish cosine-similarity NN step


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name_norm"] = df["business_name"].map(norm.normalize_name)
    df["addr_norm"] = df["business_address"].map(norm.normalize_address)
    df["name_tokens"] = df["name_norm"].map(norm.tokenize)
    df["addr_tokens"] = df["addr_norm"].map(norm.tokenize)
    df["name_first_tok"] = df["name_tokens"].map(lambda t: t[0] if t else "")
    df["name_prefix"] = df["name_norm"].str.slice(0, 4)
    df["postal_codes"] = df["addr_norm"].map(norm.postal_code_candidates)
    df["numeric_tokens"] = df["addr_norm"].map(norm.numeric_tokens) + df["name_norm"].map(
        norm.numeric_tokens
    )
    df["street_number"] = df["addr_norm"].map(norm.street_number_candidate)
    return df


def _pairs_from_key(
    s1_df: pd.DataFrame, other_df: pd.DataFrame, key_col: str, is_list: bool,
    max_block_size: int = MAX_BLOCK_SIZE,
) -> pd.DataFrame:
    """Vectorized equi-join blocking on a scalar or list-valued key column."""
    left = s1_df[["entity_id", key_col]].rename(columns={"entity_id": "s1_id"})
    right = other_df[["entity_id", key_col]].rename(columns={"entity_id": "other_id"})
    if is_list:
        left = left.explode(key_col)
        right = right.explode(key_col)
    left = left[left[key_col].astype(bool)]
    right = right[right[key_col].astype(bool)]
    if left.empty or right.empty:
        return pd.DataFrame(columns=["s1_id", "other_id"])

    # Drop overly common keys on either side (not discriminative, blows up pairs)
    left_counts = left[key_col].value_counts()
    right_counts = right[key_col].value_counts()
    ok_keys = set(left_counts[left_counts <= max_block_size].index) & set(
        right_counts[right_counts <= max_block_size].index
    )
    if not ok_keys:
        return pd.DataFrame(columns=["s1_id", "other_id"])
    left = left[left[key_col].isin(ok_keys)]
    right = right[right[key_col].isin(ok_keys)]

    merged = left.merge(right, on=key_col, how="inner")
    return merged[["s1_id", "other_id"]].drop_duplicates()


def _tfidf_nn_pairs(
    s1_df: pd.DataFrame, other_df: pd.DataFrame, n_neighbors: int
) -> pd.DataFrame:
    """Character n-gram TF-IDF nearest-neighbor blocking.

    To stay tractable on large partitions, this is computed bucket-by-bucket
    on (name_prefix) sub-buckets rather than globally: cosine similarity is
    computed only between an S1 row and the S2/S3 rows sharing its 4-char
    name prefix bucket. Buckets larger than TFIDF_BUCKET_MAX are skipped
    (other blocking rules already provide a recall baseline for those very
    common-prefix names, e.g. "the ", "inc ").
    """
    results = []
    other_by_bucket = {k: v for k, v in other_df.groupby("name_prefix")}
    for prefix, s1_bucket in s1_df.groupby("name_prefix"):
        other_bucket = other_by_bucket.get(prefix)
        if other_bucket is None or other_bucket.empty:
            continue
        n1, n2 = len(s1_bucket), len(other_bucket)
        if n1 * n2 > TFIDF_BUCKET_MAX * TFIDF_BUCKET_MAX or n1 == 0 or n2 == 0:
            continue
        corpus = pd.concat([s1_bucket["name_norm"], other_bucket["name_norm"]])
        if corpus.str.len().sum() == 0:
            continue
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NAME_NGRAM_RANGE, min_df=1)
            X = vec.fit_transform(corpus)
        except ValueError:
            continue
        X1 = X[:n1]
        X2 = X[n1:]
        if X2.shape[0] == 0:
            continue
        sim = X1 @ X2.T  # sparse cosine-ish similarity (rows are l2-normalized by TfidfVectorizer)
        sim = sim.tocsr()
        k = min(n_neighbors, n2)
        s1_ids = s1_bucket["entity_id"].to_numpy()
        other_ids = other_bucket["entity_id"].to_numpy()
        for row_i in range(sim.shape[0]):
            row = sim.getrow(row_i)
            if row.nnz == 0:
                continue
            top_idx = np.argsort(row.data)[-k:]
            cols = row.indices[top_idx]
            for c in cols:
                results.append((s1_ids[row_i], other_ids[c]))
    if not results:
        return pd.DataFrame(columns=["s1_id", "other_id"])
    return pd.DataFrame(results, columns=["s1_id", "other_id"]).drop_duplicates()


def generate_candidates_for_country(
    s1_df: pd.DataFrame, other_df: pd.DataFrame, cfg
) -> pd.DataFrame:
    """Union of all blocking rules for one (country, source-side) partition."""
    if s1_df.empty or other_df.empty:
        return pd.DataFrame(columns=["s1_id", "other_id"])

    pair_frames = [
        _pairs_from_key(s1_df, other_df, "name_norm", is_list=False),          # rule 1
        _pairs_from_key(s1_df, other_df, "name_prefix", is_list=False),        # rule 2
        _pairs_from_key(s1_df, other_df, "name_first_tok", is_list=False),     # rule 3
        _pairs_from_key(s1_df, other_df, "postal_codes", is_list=True),        # rule 4a
        _pairs_from_key(s1_df, other_df, "numeric_tokens", is_list=True,
                         max_block_size=HIGH_FREQUNCY_BLOCK_CAP),               # rule 4b
        _tfidf_nn_pairs(s1_df, other_df, cfg.n_neighbors),                     # rule 5
        _address_token_overlap_pairs(s1_df, other_df, cfg),                   # rule 6
    ]
    pairs = pd.concat(pair_frames, ignore_index=True).drop_duplicates()
    return pairs


def _address_token_overlap_pairs(s1_df: pd.DataFrame, other_df: pd.DataFrame, cfg) -> pd.DataFrame:
    def keep_long_tokens(tokens):
        return [t for t in tokens if len(t) >= cfg.min_token_len_for_index and not t.isdigit()]

    s1_tmp = s1_df[["entity_id", "addr_tokens"]].copy()
    other_tmp = other_df[["entity_id", "addr_tokens"]].copy()
    s1_tmp["addr_tokens"] = s1_tmp["addr_tokens"].map(keep_long_tokens)
    other_tmp["addr_tokens"] = other_tmp["addr_tokens"].map(keep_long_tokens)
    return _pairs_from_key(
        s1_tmp, other_tmp, "addr_tokens", is_list=True, max_block_size=HIGH_FREQUNCY_BLOCK_CAP
    )


def _quick_name_score(name_a: str, name_b: str) -> float:
    """Cheap ranking score used only to cap candidate counts, not for the model."""
    if not name_a or not name_b:
        return 0.0
    set_a, set_b = set(name_a.split(" ")), set(name_b.split(" "))
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def cap_candidates_per_s1(
    pairs_s2: pd.DataFrame,
    pairs_s3: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates: int,
) -> Dict[str, List[str]]:
    """Union S2+S3 pairs per s1 id, rank by a cheap name-overlap score, cap size."""
    all_pairs = pd.concat(
        [
            pairs_s2.rename(columns={"other_id": "cand_id"}),
            pairs_s3.rename(columns={"other_id": "cand_id"}),
        ],
        ignore_index=True,
    ).drop_duplicates()

    s1_name_map = s1_df.set_index("entity_id")["name_norm"].to_dict()
    cand_name_map = pd.concat(
        [s2_df.set_index("entity_id")["name_norm"], s3_df.set_index("entity_id")["name_norm"]]
    ).to_dict()

    out: Dict[str, List[str]] = {}
    for s1_id, group in all_pairs.groupby("s1_id"):
        cand_ids = group["cand_id"].tolist()
        if len(cand_ids) > max_candidates:
            name_a = s1_name_map.get(s1_id, "")
            scored = sorted(
                cand_ids,
                key=lambda cid: _quick_name_score(name_a, cand_name_map.get(cid, "")),
                reverse=True,
            )
            cand_ids = scored[:max_candidates]
        out[s1_id] = cand_ids
    return out