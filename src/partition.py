"""Chunked country-partitioning: split huge TSVs into small per-country files
on disk WITHOUT ever loading the full file into memory. This is what lets
run_pipeline process one country at a time instead of holding millions of
rows across all countries simultaneously.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

CHUNKSIZE = 300_000


def _safe_country_dir(dest_root: Path, country: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in str(country)) or "unknown"
    d = dest_root / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def partition_records_by_country(src_path: Path, dest_root: Path, out_filename: str):
    """Stream src_path in chunks, appending each row to dest_root/<country>/out_filename."""
    for chunk in pd.read_csv(src_path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNKSIZE):
        for country, sub in chunk.groupby("country", dropna=False):
            out_path = _safe_country_dir(dest_root, country) / out_filename
            sub.to_csv(out_path, sep="\t", index=False, mode="a", header=not out_path.exists())


def build_id_country_map(src_path: Path) -> Dict[str, str]:
    """Read only entity_id + country (2 columns) in chunks -- much lighter than
    loading the full record file. Used to partition ground_truth by the
    country of its source1_entity_id."""
    id_country: Dict[str, str] = {}
    for chunk in pd.read_csv(
        src_path, sep="\t", dtype=str, keep_default_na=False,
        usecols=["entity_id", "country"], chunksize=CHUNKSIZE,
    ):
        id_country.update(dict(zip(chunk["entity_id"], chunk["country"])))
    return id_country


def partition_ground_truth_by_country(
    gt_path: Path, id_country: Dict[str, str], dest_root: Path, out_filename: str = "train_ground_truth.tsv"
):
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNKSIZE):
        chunk["_country"] = chunk["source1_entity_id"].map(id_country)
        for country, sub in chunk.groupby("_country", dropna=False):
            sub = sub.drop(columns=["_country"])
            out_path = _safe_country_dir(dest_root, country) / out_filename
            sub.to_csv(out_path, sep="\t", index=False, mode="a", header=not out_path.exists())


def partition_train_dir(train_dir: Path, dest_root: Path, force: bool = False):
    """Partition train_source1/2/3.tsv + train_ground_truth.tsv by country."""
    marker = dest_root / "_DONE_train"
    if marker.exists() and not force:
        return
    if dest_root.exists() and force:
        import shutil
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    partition_records_by_country(train_dir / "train_source1.tsv", dest_root, "train_source1.tsv")
    partition_records_by_country(train_dir / "train_source2.tsv", dest_root, "train_source2.tsv")
    partition_records_by_country(train_dir / "train_source3.tsv", dest_root, "train_source3.tsv")

    id_country = build_id_country_map(train_dir / "train_source1.tsv")
    partition_ground_truth_by_country(train_dir / "train_ground_truth.tsv", id_country, dest_root)
    del id_country

    marker.write_text("done")


def partition_test_dir(test_dir: Path, dest_root: Path, force: bool = False):
    """Partition test_source1/2/3.tsv by country."""
    marker = dest_root / "_DONE_test"
    if marker.exists() and not force:
        return
    if dest_root.exists() and force:
        import shutil
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    partition_records_by_country(test_dir / "test_source1.tsv", dest_root, "test_source1.tsv")
    partition_records_by_country(test_dir / "test_source2.tsv", dest_root, "test_source2.tsv")
    partition_records_by_country(test_dir / "test_source3.tsv", dest_root, "test_source3.tsv")

    marker.write_text("done")


def list_country_dirs(dest_root: Path):
    return sorted(d for d in dest_root.iterdir() if d.is_dir())