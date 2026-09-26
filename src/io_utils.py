"""IO helpers: loading and validating the TSV inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

REQUIRED_RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


class SchemaError(ValueError):
    pass


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def _validate_columns(df: pd.DataFrame, required: List[str], path: Path):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{path} is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def load_records(path: Path, expected_prefix: str) -> pd.DataFrame:
    """Load a source records file (source1/source2/source3) and validate schema."""
    df = _read_tsv(path)
    _validate_columns(df, REQUIRED_RECORD_COLUMNS, path)
    df = df[REQUIRED_RECORD_COLUMNS].copy()

    # Sanity-check entity_id prefixes (warn, don't crash, in case of edge cases)
    bad_prefix = ~df["entity_id"].astype(str).str.startswith(expected_prefix)
    if bad_prefix.any():
        n_bad = int(bad_prefix.sum())
        print(
            f"[io_utils] WARNING: {n_bad} rows in {path} do not start with "
            f"expected prefix '{expected_prefix}'."
        )

    df = df.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    return df


def load_ground_truth(path: Path) -> Dict[str, List[str]]:
    """Load train_ground_truth.tsv into {source1_entity_id: [matched ids]}."""
    df = _read_tsv(path)
    _validate_columns(df, REQUIRED_GT_COLUMNS, path)

    gt: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw = str(row["matched_entity_ids"]).strip()
        if raw == "" or raw.lower() == "nan":
            matched: List[str] = []
        else:
            matched = [x.strip() for x in raw.split(",") if x.strip() != ""]
        gt[s1_id] = matched
    return gt


def write_matching_results(path: Path, results: Dict[str, List[str]], s1_order: List[str]):
    """Write matching_results.tsv preserving s1_order, one row per source1 entity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id in s1_order:
        matched = results.get(s1_id, [])
        # de-duplicate while preserving order
        seen = set()
        clean = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                clean.append(m)
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(clean)})
    out = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    out.to_csv(path, sep="\t", index=False)


def write_candidate_pairs(path: Path, candidates: Dict[str, List[str]], s1_order: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id in s1_order:
        cand = candidates.get(s1_id, [])
        seen = set()
        clean = []
        for c in cand:
            if c not in seen:
                seen.add(c)
                clean.append(c)
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(clean)})
    out = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    out.to_csv(path, sep="\t", index=False)


def read_candidate_pairs(path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"])
        raw = str(row["candidate_entity_ids"]).strip()
        out[s1_id] = [x for x in raw.split(",") if x] if raw else []
    return out
