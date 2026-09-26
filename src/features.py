"""Pairwise similarity feature engineering.

All TF-IDF vectorizers / IDF weight tables used here must be fit ONCE on
training data (see `fit_feature_transformers`) and then reused unchanged
for validation and test prediction -- `compute_features` never re-fits
anything, it only calls `.transform(...)`.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

CHAR_NGRAM_RANGE = (2, 4)

FEATURE_COLUMNS = [
    # name features
    "name_exact_match",
    "name_ratio",
    "name_wratio",
    "name_edit_sim",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_token_jaccard",
    "name_tfidf_cosine",
    "name_containment",
    "name_len_diff",
    "name_shared_rare_tokens",
    # address features
    "addr_exact_match",
    "addr_edit_sim",
    "addr_token_overlap",
    "addr_token_jaccard",
    "addr_tfidf_cosine",
    "addr_shared_numeric_tokens",
    "addr_postal_agreement",
    "addr_street_number_agreement",
    "addr_len_diff",
    "addr_s1_missing",
    "addr_cand_missing",
    # other
    "country_exact_match",
    "is_source2",
    "combined_name_addr_score",
    "name_addr_disagreement",
    "name_s1_missing",
    "name_cand_missing",
]


class FeatureTransformers:
    def __init__(self, name_vec: TfidfVectorizer, addr_vec: TfidfVectorizer, rare_idf: Dict[str, float]):
        self.name_vec = name_vec
        self.addr_vec = addr_vec
        self.rare_idf = rare_idf


def fit_feature_transformers(train_records: pd.DataFrame) -> FeatureTransformers:
    """Fit TF-IDF vectorizers and a word-level IDF table on TRAINING records only.

    `train_records` must have `name_norm` and `addr_norm` columns and should
    include rows from source1 + source2 + source3 of the *training* split
    only (never test data), per the "fit on train, reuse unchanged" rule.
    """
    name_corpus = train_records["name_norm"].fillna("")
    addr_corpus = train_records["addr_norm"].fillna("")

    name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAM_RANGE, min_df=2)
    name_vec.fit(name_corpus[name_corpus.str.len() > 0])

    addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAM_RANGE, min_df=2)
    addr_vec.fit(addr_corpus[addr_corpus.str.len() > 0])

    # word-level IDF table for "shared rare token" scoring
    word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=1)
    word_vec.fit(name_corpus[name_corpus.str.len() > 0])
    rare_idf = dict(zip(word_vec.get_feature_names_out(), word_vec.idf_))

    return FeatureTransformers(name_vec, addr_vec, rare_idf)


def _prep_pairs(
    pairs: pd.DataFrame, s1_records: pd.DataFrame, cand_records: pd.DataFrame
) -> pd.DataFrame:
    """Join normalized fields of both sides onto the (s1_id, cand_id) pairs."""
    s1_cols = [
        "entity_id", "name_norm", "addr_norm", "name_tokens", "addr_tokens",
        "postal_codes", "numeric_tokens", "street_number", "country",
    ]
    df = pairs.merge(
        s1_records[s1_cols].rename(columns={c: f"{c}_s1" for c in s1_cols if c != "entity_id"}),
        left_on="s1_id", right_on="entity_id", how="left",
    ).drop(columns=["entity_id"])
    cand_cols = s1_cols
    df = df.merge(
        cand_records[cand_cols].rename(columns={c: f"{c}_cand" for c in cand_cols if c != "entity_id"}),
        left_on="cand_id", right_on="entity_id", how="left",
    ).drop(columns=["entity_id"])
    return df


def compute_features(
    pairs: pd.DataFrame,
    s1_records: pd.DataFrame,
    cand_records: pd.DataFrame,
    transformers: FeatureTransformers,
) -> pd.DataFrame:
    """Compute the full feature matrix for a set of (s1_id, cand_id) pairs.

    `cand_records` should already be tagged with a boolean `is_source2` column
    (True for source2 rows, False for source3 rows) so the source-pair
    indicator can be derived.
    """
    if pairs.empty:
        return pd.DataFrame(columns=["s1_id", "cand_id"] + FEATURE_COLUMNS)

    df = _prep_pairs(pairs, s1_records, cand_records)

    name_a = df["name_norm_s1"].fillna("")
    name_b = df["name_norm_cand"].fillna("")
    addr_a = df["addr_norm_s1"].fillna("")
    addr_b = df["addr_norm_cand"].fillna("")

    n = len(df)
    feats: Dict[str, np.ndarray] = {}

    # ---- name features ----
    feats["name_exact_match"] = (name_a == name_b).to_numpy(dtype=float) * (name_a != "").to_numpy(dtype=float)
    feats["name_ratio"] = np.array(
        [fuzz.ratio(a, b) / 100.0 for a, b in zip(name_a, name_b)]
    )
    feats["name_wratio"] = np.array(
        [fuzz.WRatio(a, b) / 100.0 for a, b in zip(name_a, name_b)]
    )
    feats["name_edit_sim"] = np.array(
        [Levenshtein.normalized_similarity(a, b) for a, b in zip(name_a, name_b)]
    )
    feats["name_token_set_ratio"] = np.array(
        [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(name_a, name_b)]
    )
    feats["name_token_sort_ratio"] = np.array(
        [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(name_a, name_b)]
    )

    name_tok_a = df["name_tokens_s1"].apply(lambda x: set(x) if isinstance(x, list) else set())
    name_tok_b = df["name_tokens_cand"].apply(lambda x: set(x) if isinstance(x, list) else set())
    jacc, contain, shared_rare = [], [], []
    rare_idf = transformers.rare_idf
    for sa, sb in zip(name_tok_a, name_tok_b):
        inter = sa & sb
        union = sa | sb
        jacc.append(len(inter) / len(union) if union else 0.0)
        m = min(len(sa), len(sb))
        contain.append(len(inter) / m if m else 0.0)
        shared_rare.append(sum(rare_idf.get(t, 0.0) for t in inter))
    feats["name_token_jaccard"] = np.array(jacc)
    feats["name_containment"] = np.array(contain)
    feats["name_shared_rare_tokens"] = np.array(shared_rare)

    Xa = transformers.name_vec.transform(name_a)
    Xb = transformers.name_vec.transform(name_b)
    feats["name_tfidf_cosine"] = np.asarray(Xa.multiply(Xb).sum(axis=1)).ravel()

    feats["name_len_diff"] = np.abs(name_a.str.len().to_numpy() - name_b.str.len().to_numpy()).astype(float)
    feats["name_s1_missing"] = (name_a == "").to_numpy(dtype=float)
    feats["name_cand_missing"] = (name_b == "").to_numpy(dtype=float)

    # ---- address features ----
    feats["addr_exact_match"] = (addr_a == addr_b).to_numpy(dtype=float) * (addr_a != "").to_numpy(dtype=float)
    feats["addr_edit_sim"] = np.array(
        [Levenshtein.normalized_similarity(a, b) for a, b in zip(addr_a, addr_b)]
    )

    addr_tok_a = df["addr_tokens_s1"].apply(lambda x: set(x) if isinstance(x, list) else set())
    addr_tok_b = df["addr_tokens_cand"].apply(lambda x: set(x) if isinstance(x, list) else set())
    ov, ajacc = [], []
    for sa, sb in zip(addr_tok_a, addr_tok_b):
        inter = sa & sb
        union = sa | sb
        ov.append(len(inter))
        ajacc.append(len(inter) / len(union) if union else 0.0)
    feats["addr_token_overlap"] = np.array(ov, dtype=float)
    feats["addr_token_jaccard"] = np.array(ajacc)

    Xa2 = transformers.addr_vec.transform(addr_a)
    Xb2 = transformers.addr_vec.transform(addr_b)
    feats["addr_tfidf_cosine"] = np.asarray(Xa2.multiply(Xb2).sum(axis=1)).ravel()

    num_a = df["numeric_tokens_s1"].apply(lambda x: set(x) if isinstance(x, list) else set())
    num_b = df["numeric_tokens_cand"].apply(lambda x: set(x) if isinstance(x, list) else set())
    feats["addr_shared_numeric_tokens"] = np.array(
        [len(a & b) for a, b in zip(num_a, num_b)], dtype=float
    )

    postal_a = df["postal_codes_s1"].apply(lambda x: set(x) if isinstance(x, list) else set())
    postal_b = df["postal_codes_cand"].apply(lambda x: set(x) if isinstance(x, list) else set())
    feats["addr_postal_agreement"] = np.array(
        [1.0 if (a and b and (a & b)) else 0.0 for a, b in zip(postal_a, postal_b)]
    )

    street_a = df["street_number_s1"].fillna("")
    street_b = df["street_number_cand"].fillna("")
    feats["addr_street_number_agreement"] = np.array(
        [1.0 if (a and b and a == b) else 0.0 for a, b in zip(street_a, street_b)]
    )

    feats["addr_len_diff"] = np.abs(addr_a.str.len().to_numpy() - addr_b.str.len().to_numpy()).astype(float)
    feats["addr_s1_missing"] = (addr_a == "").to_numpy(dtype=float)
    feats["addr_cand_missing"] = (addr_b == "").to_numpy(dtype=float)

    # ---- other / combined features ----
    country_a = df["country_s1"].fillna("")
    country_b = df["country_cand"].fillna("")
    feats["country_exact_match"] = (country_a.str.lower() == country_b.str.lower()).to_numpy(dtype=float)

    feats["is_source2"] = df["cand_id"].astype(str).str.startswith("S2-").to_numpy(dtype=float)

    feats["combined_name_addr_score"] = 0.6 * feats["name_token_set_ratio"] + 0.4 * feats["addr_edit_sim"]
    feats["name_addr_disagreement"] = np.abs(feats["name_token_set_ratio"] - feats["addr_edit_sim"])

    out = pd.DataFrame({"s1_id": df["s1_id"].to_numpy(), "cand_id": df["cand_id"].to_numpy()})
    for col in FEATURE_COLUMNS:
        out[col] = feats[col]
    return out


def compute_features_parallel(
    pairs: pd.DataFrame,
    s1_records: pd.DataFrame,
    cand_records: pd.DataFrame,
    transformers: FeatureTransformers,
    n_jobs: int = -1,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Chunk `pairs` and compute features per chunk in parallel processes.

    Only the (small) required columns of s1_records/cand_records are shipped
    to each worker (not the full dataframes) to keep IPC overhead down.
    Safe to use for both moderate and very large (tens of millions of rows)
    pair sets -- for small inputs it degrades gracefully to a single chunk.
    """
    if pairs.empty:
        return pd.DataFrame(columns=["s1_id", "cand_id"] + FEATURE_COLUMNS)

    keep_cols = [
        "entity_id", "name_norm", "addr_norm", "name_tokens", "addr_tokens",
        "postal_codes", "numeric_tokens", "street_number", "country",
    ]
    s1_slim = s1_records[s1_records["entity_id"].isin(set(pairs["s1_id"]))][keep_cols]
    cand_slim = cand_records[cand_records["entity_id"].isin(set(pairs["cand_id"]))][keep_cols]

    pairs = pairs.reset_index(drop=True)
    chunks = [pairs.iloc[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_features)(chunk, s1_slim, cand_slim, transformers)
        for chunk in chunks
        if len(chunk) > 0
    )
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["s1_id", "cand_id"] + FEATURE_COLUMNS
    )
