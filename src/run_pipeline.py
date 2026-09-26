"""End-to-end pipeline: train -> tune threshold -> predict on test -> validate.

MEMORY DESIGN: this pipeline never loads the full train or test dataset into
memory at once. It first partitions each huge TSV by country on disk (via
chunked/streaming reads, src/partition.py), then processes ONE COUNTRY AT A
TIME -- loading, normalizing, blocking, and feature-computing just that
country's (much smaller) slice before discarding it and moving to the next.
This is what makes a multi-million-row dataset feasible on a 16GB machine.

Run as:
    python -m src.run_pipeline --train-dir dataset/train --test-dir dataset/test --output-dir output

Or split into two separate process runs (extra safety -- e.g. so a crash
during prediction doesn't require re-training):
    python -m src.run_pipeline --phase train   --train-dir dataset/train --output-dir output
    python -m src.run_pipeline --phase predict --test-dir dataset/test --output-dir output
"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import joblib
import pandas as pd

from . import blocking, evaluation, io_utils, model as model_mod, partition, predict
from .config import Config, build_arg_parser, config_from_args
from .features import compute_features_parallel, fit_feature_transformers
from . import normalization as norm


def log(msg: str):
    print(f"[run_pipeline] {msg}", flush=True)


# --------------------------------------------------------------------------
# Lightweight streaming pass: fit TF-IDF/IDF transformers on TRAIN data only,
# without ever holding full normalized dataframes for every country at once.
# --------------------------------------------------------------------------
def fit_transformers_streaming(country_dirs: List[Path]) -> "object":
    name_parts, addr_parts = [], []
    for cdir in country_dirs:
        for fname in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
            fpath = cdir / fname
            if not fpath.exists():
                continue
            df = pd.read_csv(
                fpath, sep="\t", dtype=str, keep_default_na=False,
                usecols=["business_name", "business_address"],
            )
            name_parts.append(df["business_name"].map(norm.normalize_name))
            addr_parts.append(df["business_address"].map(norm.normalize_address))
            del df
        gc.collect()

    corpus = pd.DataFrame({
        "name_norm": pd.concat(name_parts, ignore_index=True),
        "addr_norm": pd.concat(addr_parts, ignore_index=True),
    })
    del name_parts, addr_parts
    transformers = fit_feature_transformers(corpus)
    del corpus
    gc.collect()
    return transformers


def validate_outputs(matching_path: Path, candidate_path: Path, test_source1_ids: List[str]):
    """Lightweight internal validation of the required invariants."""
    errors = []
    mr = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    cp = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)

    if set(mr["source1_entity_id"]) != set(test_source1_ids):
        errors.append("matching_results.tsv does not cover exactly the test source1 ids")
    if mr["source1_entity_id"].duplicated().any():
        errors.append("matching_results.tsv has duplicate source1_entity_id rows")
    if set(cp["source1_entity_id"]) != set(test_source1_ids):
        errors.append("candidate_pairs.tsv does not cover exactly the test source1 ids")
    if cp["source1_entity_id"].duplicated().any():
        errors.append("candidate_pairs.tsv has duplicate source1_entity_id rows")

    cand_map = {}
    for _, row in cp.iterrows():
        raw = row["candidate_entity_ids"]
        cand_map[row["source1_entity_id"]] = set(raw.split(",")) if raw else set()

    for _, row in mr.iterrows():
        raw = row["matched_entity_ids"]
        matched = [m for m in raw.split(",") if m] if raw else []
        if len(matched) != len(set(matched)):
            errors.append(f"duplicate matched ids for {row['source1_entity_id']}")
        allowed = cand_map.get(row["source1_entity_id"], set())
        bad = [m for m in matched if m not in allowed]
        if bad:
            errors.append(f"{row['source1_entity_id']}: matched ids not in candidate set: {bad}")
        bad_prefix = [m for m in matched if not (m.startswith("S2-") or m.startswith("S3-"))]
        if bad_prefix:
            errors.append(f"{row['source1_entity_id']}: invalid id prefix: {bad_prefix}")

    if errors:
        log("VALIDATION FAILED:")
        for e in errors[:50]:
            log(f"  - {e}")
        raise AssertionError(f"{len(errors)} validation error(s); see log above.")
    log("Internal validation passed: ids unique/complete, all matches in candidate set.")


def _country_candidates(s1n, s2n, s3n, cfg) -> Dict[str, List[str]]:
    """Blocking + capping for ONE already-loaded country partition."""
    pairs_s2 = blocking.generate_candidates_for_country(s1n, s2n, cfg) if not s2n.empty else pd.DataFrame(columns=["s1_id", "other_id"])
    pairs_s3 = blocking.generate_candidates_for_country(s1n, s3n, cfg) if not s3n.empty else pd.DataFrame(columns=["s1_id", "other_id"])
    log(f"    raw pairs s2={len(pairs_s2)} s3={len(pairs_s3)}")
    candidates = blocking.cap_candidates_per_s1(pairs_s2, pairs_s3, s1n, s2n, s3n, cfg.max_candidates)
    del pairs_s2, pairs_s3
    return candidates


# --------------------------------------------------------------------------
# TRAIN PHASE -- processes one country at a time, never holding the full
# train dataset in memory.
# --------------------------------------------------------------------------
def run_train_phase(cfg: Config) -> dict:
    log("=== TRAIN PHASE ===")
    log("Partitioning training data by country (streamed, cached on disk)...")
    partition.partition_train_dir(cfg.train_dir, cfg.partitioned_train_dir, force=cfg.force_repartition)
    country_dirs = partition.list_country_dirs(cfg.partitioned_train_dir)
    log(f"  countries found: {[d.name for d in country_dirs]}")

    log("Fitting feature transformers on training data only (streaming pass)...")
    transformers = fit_transformers_streaming(country_dirs)

    X_train_parts, X_val_parts = [], []
    gt_sets: Dict[str, Set[str]] = {}
    val_ids_all: List[str] = []
    total_candidate_pairs = 0
    total_found = 0
    total_true = 0
    total_s1 = 0
    total_s2 = 0
    total_s3 = 0

    for cdir in country_dirs:
        country = cdir.name
        log(f"  [{country}] loading...")
        s1_path = cdir / "train_source1.tsv"
        if not s1_path.exists():
            continue
        s1 = io_utils.load_records(s1_path, "S1-")
        s2 = io_utils.load_records(cdir / "train_source2.tsv", "S2-") if (cdir / "train_source2.tsv").exists() else pd.DataFrame(columns=io_utils.REQUIRED_RECORD_COLUMNS)
        s3 = io_utils.load_records(cdir / "train_source3.tsv", "S3-") if (cdir / "train_source3.tsv").exists() else pd.DataFrame(columns=io_utils.REQUIRED_RECORD_COLUMNS)
        gt_path = cdir / "train_ground_truth.tsv"
        gt = io_utils.load_ground_truth(gt_path) if gt_path.exists() else {}
        total_s1 += len(s1); total_s2 += len(s2); total_s3 += len(s3)
        log(f"  [{country}] s1={len(s1)} s2={len(s2)} s3={len(s3)} gt_rows={len(gt)}")

        s1n = blocking.add_normalized_columns(s1)
        s2n = blocking.add_normalized_columns(s2)
        s3n = blocking.add_normalized_columns(s3)
        del s1, s2, s3

        candidates = _country_candidates(s1n, s2n, s3n, cfg)
        n_pairs = sum(len(v) for v in candidates.values())
        total_candidate_pairs += n_pairs
        rec, found, tot = evaluation.candidate_recall(candidates, gt)
        total_found += found
        total_true += tot
        log(f"  [{country}] candidates={n_pairs} recall={rec:.4f} ({found}/{tot})")

        gt_sets.update({k: set(v) for k, v in gt.items()})

        labels = evaluation.build_pairwise_labels(candidates, gt)
        s1_ids = list(candidates.keys())
        tr_ids, va_ids = evaluation.group_split(s1_ids, cfg.val_fraction, cfg.random_seed)
        tr_set, va_set = set(tr_ids), set(va_ids)
        val_ids_all.extend(va_ids)

        labels_tr = labels[labels["s1_id"].isin(tr_set)]
        labels_va = labels[labels["s1_id"].isin(va_set)]
        del labels, candidates

        cand_n = pd.concat([s2n, s3n], ignore_index=True)
        if not labels_tr.empty:
            X_tr = compute_features_parallel(
                labels_tr[["s1_id", "cand_id"]], s1n, cand_n, transformers, cfg.n_jobs, cfg.feature_chunk_size
            )
            X_tr = X_tr.merge(labels_tr[["s1_id", "cand_id", "label"]], on=["s1_id", "cand_id"])
            X_train_parts.append(X_tr)
        if not labels_va.empty:
            X_va = compute_features_parallel(
                labels_va[["s1_id", "cand_id"]], s1n, cand_n, transformers, cfg.n_jobs, cfg.feature_chunk_size
            )
            X_va = X_va.merge(labels_va[["s1_id", "cand_id", "label"]], on=["s1_id", "cand_id"])
            X_val_parts.append(X_va)

        del s1n, s2n, s3n, cand_n, labels_tr, labels_va
        gc.collect()

    log(f"TOTAL across countries: s1={total_s1} s2={total_s2} s3={total_s3} "
        f"candidate_pairs={total_candidate_pairs}")
    rec_overall = total_found / total_true if total_true else 1.0
    possible_pairs = total_s1 * (total_s2 + total_s3)
    reduction_ratio = 1 - (total_candidate_pairs / possible_pairs) if possible_pairs else 0.0
    log(f"  overall candidate recall: {rec_overall:.4f} ({total_found}/{total_true})")
    log(f"  overall candidate reduction ratio: {reduction_ratio:.6f}")

    X_train = pd.concat(X_train_parts, ignore_index=True) if X_train_parts else pd.DataFrame()
    X_val = pd.concat(X_val_parts, ignore_index=True) if X_val_parts else pd.DataFrame()
    del X_train_parts, X_val_parts
    gc.collect()
    log(f"  final train_pairs={len(X_train)} val_pairs={len(X_val)} "
        f"positives_train={int(X_train['label'].sum()) if len(X_train) else 0}")

    log("Training model on TRAIN split...")
    clf = model_mod.train_model(X_train, X_train["label"].to_numpy(), cfg.random_seed)

    log("Scoring VAL split and tuning threshold for macro F0.5...")
    val_scored = X_val[["s1_id", "cand_id"]].copy()
    val_scored["score"] = model_mod.predict_proba(clf, X_val)

    if cfg.fixed_threshold is not None:
        best_threshold = cfg.fixed_threshold
        threshold_table = pd.DataFrame([{"threshold": best_threshold, "note": "fixed by user"}])
    else:
        best_threshold, threshold_table = evaluation.tune_threshold(
            val_scored, gt_sets, val_ids_all, cfg.thresholds
        )
    log(f"  best threshold = {best_threshold}")

    val_preds = evaluation.predictions_from_scores(val_scored, best_threshold)
    val_metrics = evaluation.macro_f_beta(val_preds, gt_sets, val_ids_all, beta=0.5)
    singleton_metrics = evaluation.singleton_performance(val_preds, gt_sets, val_ids_all)
    log(f"  VAL metrics: {val_metrics}")
    log(f"  VAL singleton metrics: {singleton_metrics}")

    log("Retraining final model on all labeled train+val pairs...")
    X_all = pd.concat([X_train, X_val], ignore_index=True)
    final_clf = model_mod.train_model(X_all, X_all["label"].to_numpy(), cfg.random_seed)
    del X_train, X_val, X_all
    gc.collect()

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    bundle = {"model": final_clf, "transformers": transformers, "threshold": best_threshold}
    joblib.dump(bundle, cfg.models_dir / "model_bundle.joblib")
    log(f"  Saved model bundle to {cfg.models_dir / 'model_bundle.joblib'}")

    train_report = {
        "best_threshold": best_threshold,
        "val_metrics": val_metrics,
        "val_singleton_metrics": singleton_metrics,
        "train_candidate_recall": rec_overall,
        "train_candidate_reduction_ratio": reduction_ratio,
        "n_train_candidate_pairs": total_candidate_pairs,
    }
    (cfg.reports_dir / "train_metrics.json").write_text(json.dumps(train_report, indent=2, default=str))
    threshold_table.to_csv(cfg.reports_dir / "threshold_tuning.tsv", sep="\t", index=False)
    log("=== TRAIN PHASE DONE ===")
    log(json.dumps(train_report, indent=2, default=str))
    return bundle


# --------------------------------------------------------------------------
# PREDICT PHASE -- same per-country streaming, scores against the saved
# model bundle, writes outputs in the ORIGINAL test_source1.tsv row order.
# --------------------------------------------------------------------------
def run_predict_phase(cfg: Config, bundle: Optional[dict] = None) -> dict:
    log("=== PREDICT PHASE ===")
    if bundle is None:
        bundle_path = cfg.models_dir / "model_bundle.joblib"
        log(f"Loading model bundle from {bundle_path}...")
        bundle = joblib.load(bundle_path)
    final_clf = bundle["model"]
    transformers = bundle["transformers"]
    best_threshold = bundle["threshold"]

    log("Partitioning test data by country (streamed, cached on disk)...")
    partition.partition_test_dir(cfg.test_dir, cfg.partitioned_test_dir, force=cfg.force_repartition)
    country_dirs = partition.list_country_dirs(cfg.partitioned_test_dir)
    log(f"  countries found: {[d.name for d in country_dirs]}")

    candidates_all: Dict[str, List[str]] = {}
    matches_all: Dict[str, List[str]] = {}
    total_candidate_pairs = 0

    for cdir in country_dirs:
        country = cdir.name
        s1_path = cdir / "test_source1.tsv"
        if not s1_path.exists():
            continue
        log(f"  [{country}] loading...")
        s1 = io_utils.load_records(s1_path, "S1-")
        s2 = io_utils.load_records(cdir / "test_source2.tsv", "S2-") if (cdir / "test_source2.tsv").exists() else pd.DataFrame(columns=io_utils.REQUIRED_RECORD_COLUMNS)
        s3 = io_utils.load_records(cdir / "test_source3.tsv", "S3-") if (cdir / "test_source3.tsv").exists() else pd.DataFrame(columns=io_utils.REQUIRED_RECORD_COLUMNS)
        log(f"  [{country}] s1={len(s1)} s2={len(s2)} s3={len(s3)}")

        s1n = blocking.add_normalized_columns(s1)
        s2n = blocking.add_normalized_columns(s2)
        s3n = blocking.add_normalized_columns(s3)
        del s1, s2, s3

        candidates = _country_candidates(s1n, s2n, s3n, cfg)
        n_pairs = sum(len(v) for v in candidates.values())
        total_candidate_pairs += n_pairs
        log(f"  [{country}] candidates={n_pairs}")
        candidates_all.update(candidates)

        cand_n = pd.concat([s2n, s3n], ignore_index=True)
        scored = predict.score_candidates(
            candidates, s1n, cand_n, final_clf, transformers,
            n_jobs=cfg.n_jobs, chunk_size=cfg.feature_chunk_size,
        )
        country_matches = predict.apply_threshold(scored, best_threshold, list(candidates.keys()))
        matches_all.update(country_matches)

        del s1n, s2n, s3n, cand_n, candidates, scored, country_matches
        gc.collect()

    log("Determining original test source1 row order...")
    s1_order = pd.read_csv(
        cfg.test_dir / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
        usecols=["entity_id"],
    )["entity_id"].tolist()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    io_utils.write_candidate_pairs(cfg.output_dir / "candidate_pairs.tsv", candidates_all, s1_order)
    io_utils.write_matching_results(cfg.output_dir / "matching_results.tsv", matches_all, s1_order)
    del candidates_all
    gc.collect()

    validate_outputs(
        cfg.output_dir / "matching_results.tsv", cfg.output_dir / "candidate_pairs.tsv", s1_order
    )

    validator_script = Path("utils/validate_submission.py")
    if validator_script.exists():
        log("Running official validator...")
        result = subprocess.run(
            [sys.executable, str(validator_script),
             "--matching", str(cfg.output_dir / "matching_results.tsv"),
             "--candidate", str(cfg.output_dir / "candidate_pairs.tsv"),
             "--test-dir", str(cfg.test_dir)],
            capture_output=True, text=True,
        )
        log(result.stdout)
        if result.returncode != 0:
            log(result.stderr)
        validator_passed = result.returncode == 0
    else:
        validator_passed = None

    n_matched_test = sum(1 for v in matches_all.values() if v)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    predict_report = {
        "threshold_used": best_threshold,
        "n_test_candidate_pairs": total_candidate_pairs,
        "n_test_source1_entities": len(s1_order),
        "n_test_entities_with_a_match": n_matched_test,
        "validator_passed": validator_passed,
    }
    (cfg.reports_dir / "predict_metrics.json").write_text(json.dumps(predict_report, indent=2, default=str))
    log("=== PREDICT PHASE DONE ===")
    log(json.dumps(predict_report, indent=2, default=str))
    return predict_report


def main(argv=None):
    t0 = time.time()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    bundle = None
    if cfg.phase in ("train", "all"):
        bundle = run_train_phase(cfg)
    if cfg.phase in ("predict", "all"):
        run_predict_phase(cfg, bundle)

    log(f"Total elapsed: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()