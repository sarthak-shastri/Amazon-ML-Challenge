"""End-to-end pipeline: train -> tune threshold -> predict on test -> validate.

Run as:
    python -m src.run_pipeline --train-dir dataset/train --test-dir dataset/test --output-dir output
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from joblib import Parallel, delayed

from . import blocking, evaluation, io_utils, model as model_mod, predict
from .config import Config, build_arg_parser, config_from_args
from .features import fit_feature_transformers


def log(msg: str):
    print(f"[run_pipeline] {msg}", flush=True)


def generate_candidates_all_countries(
    s1_norm: pd.DataFrame, s2_norm: pd.DataFrame, s3_norm: pd.DataFrame, cfg: Config
) -> Dict[str, List[str]]:
    """Country-partitioned candidate generation, unioned back together.

    Country partitioning is a blocking optimization only (verified on the
    training ground truth: matches never cross countries). Every country
    present in the data is processed -- none are skipped or hard-coded.
    """
    countries = sorted(set(s1_norm["country"]).union(s2_norm["country"]).union(s3_norm["country"]))

    def _one_country(country):
        s1_c = s1_norm[s1_norm["country"] == country]
        s2_c = s2_norm[s2_norm["country"] == country]
        s3_c = s3_norm[s3_norm["country"] == country]
        if s1_c.empty:
            empty = pd.DataFrame(columns=["s1_id", "other_id"])
            return country, empty, empty, len(s1_c), len(s2_c), len(s3_c)
        pairs_s2 = blocking.generate_candidates_for_country(s1_c, s2_c, cfg) if not s2_c.empty else pd.DataFrame(columns=["s1_id", "other_id"])
        pairs_s3 = blocking.generate_candidates_for_country(s1_c, s3_c, cfg) if not s3_c.empty else pd.DataFrame(columns=["s1_id", "other_id"])
        return country, pairs_s2, pairs_s3, len(s1_c), len(s2_c), len(s3_c)

    # Each country partition is independent -> safe to run across processes.
    results = Parallel(n_jobs=cfg.n_jobs, backend="loky")(
        delayed(_one_country)(c) for c in countries
    )

    all_pairs_s2, all_pairs_s3 = [], []
    for country, pairs_s2, pairs_s3, n1, n2, n3 in results:
        all_pairs_s2.append(pairs_s2)
        all_pairs_s3.append(pairs_s3)
        log(f"  country={country!r}: s1={n1} s2={n2} s3={n3} "
            f"-> raw pairs s2={len(pairs_s2)} s3={len(pairs_s3)}")

    pairs_s2_df = pd.concat(all_pairs_s2, ignore_index=True) if all_pairs_s2 else pd.DataFrame(columns=["s1_id", "other_id"])
    pairs_s3_df = pd.concat(all_pairs_s3, ignore_index=True) if all_pairs_s3 else pd.DataFrame(columns=["s1_id", "other_id"])

    candidates = blocking.cap_candidates_per_s1(
        pairs_s2_df, pairs_s3_df, s1_norm, s2_norm, s3_norm, cfg.max_candidates
    )
    return candidates


def validate_outputs(
    matching_path: Path, candidate_path: Path, test_source1_ids: List[str]
):
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


def main(argv=None):
    t0 = time.time()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load ----------------
    log("Loading training data...")
    s1_train = io_utils.load_records(cfg.train_dir / "train_source1.tsv", "S1-")
    s2_train = io_utils.load_records(cfg.train_dir / "train_source2.tsv", "S2-")
    s3_train = io_utils.load_records(cfg.train_dir / "train_source3.tsv", "S3-")
    ground_truth = io_utils.load_ground_truth(cfg.train_dir / "train_ground_truth.tsv")
    log(f"  source1={len(s1_train)} source2={len(s2_train)} source3={len(s3_train)} gt_rows={len(ground_truth)}")

    log("Loading test data...")
    s1_test = io_utils.load_records(cfg.test_dir / "test_source1.tsv", "S1-")
    s2_test = io_utils.load_records(cfg.test_dir / "test_source2.tsv", "S2-")
    s3_test = io_utils.load_records(cfg.test_dir / "test_source3.tsv", "S3-")
    log(f"  source1={len(s1_test)} source2={len(s2_test)} source3={len(s3_test)}")

    # ---------------- Normalize ----------------
    log("Normalizing text...")
    s1_train_n = blocking.add_normalized_columns(s1_train)
    s2_train_n = blocking.add_normalized_columns(s2_train)
    s3_train_n = blocking.add_normalized_columns(s3_train)
    s1_test_n = blocking.add_normalized_columns(s1_test)
    s2_test_n = blocking.add_normalized_columns(s2_test)
    s3_test_n = blocking.add_normalized_columns(s3_test)

    # ---------------- Candidate generation (train) ----------------
    log("Generating TRAIN candidates (blocking)...")
    train_candidates = generate_candidates_all_countries(s1_train_n, s2_train_n, s3_train_n, cfg)
    n_pairs_train = sum(len(v) for v in train_candidates.values())
    log(f"  train candidate pairs: {n_pairs_train} across {len(train_candidates)} source1 entities")

    rec, found, total_true = evaluation.candidate_recall(train_candidates, ground_truth)
    log(f"  candidate recall on TRAIN ground truth: {rec:.4f} ({found}/{total_true})")
    possible_pairs = len(s1_train_n) * (len(s2_train_n) + len(s3_train_n))
    reduction_ratio = 1 - (n_pairs_train / possible_pairs) if possible_pairs else 0.0
    log(f"  candidate reduction ratio: {reduction_ratio:.6f}")

    # ---------------- Labels + split ----------------
    log("Building pairwise labels and entity-level train/val split...")
    labels = evaluation.build_pairwise_labels(train_candidates, ground_truth)
    s1_ids_with_candidates = list(train_candidates.keys())
    train_ids, val_ids = evaluation.group_split(s1_ids_with_candidates, cfg.val_fraction, cfg.random_seed)
    train_ids_set, val_ids_set = set(train_ids), set(val_ids)

    labels_train = labels[labels["s1_id"].isin(train_ids_set)]
    labels_val = labels[labels["s1_id"].isin(val_ids_set)]
    log(f"  train entities={len(train_ids)} val entities={len(val_ids)} "
        f"train_pairs={len(labels_train)} val_pairs={len(labels_val)} "
        f"positives_total={int(labels['label'].sum())}")

    # ---------------- Fit feature transformers (TRAIN ONLY) ----------------
    log("Fitting feature transformers on training data only...")
    corpus = pd.concat(
        [s1_train_n[["name_norm", "addr_norm"]], s2_train_n[["name_norm", "addr_norm"]],
         s3_train_n[["name_norm", "addr_norm"]]],
        ignore_index=True,
    )
    transformers = fit_feature_transformers(corpus)

    # combined candidate-side lookup tables (source2 + source3 share entity_id namespace)
    cand_train_n = pd.concat([s2_train_n, s3_train_n], ignore_index=True)
    cand_test_n = pd.concat([s2_test_n, s3_test_n], ignore_index=True)

    # ---------------- Feature computation (train/val) ----------------
    log("Computing features for train/val pairs...")
    from .features import compute_features_parallel

    pairs_train = labels_train[["s1_id", "cand_id"]]
    pairs_val = labels_val[["s1_id", "cand_id"]]
    X_train = compute_features_parallel(
        pairs_train, s1_train_n, cand_train_n, transformers, cfg.n_jobs, cfg.feature_chunk_size
    )
    X_val = compute_features_parallel(
        pairs_val, s1_train_n, cand_train_n, transformers, cfg.n_jobs, cfg.feature_chunk_size
    )
    y_train = labels_train["label"].to_numpy()
    y_val = labels_val["label"].to_numpy()
    X_train = X_train.merge(labels_train[["s1_id", "cand_id", "label"]], on=["s1_id", "cand_id"])
    X_val = X_val.merge(labels_val[["s1_id", "cand_id", "label"]], on=["s1_id", "cand_id"])

    # ---------------- Train (initial) + tune threshold on val ----------------
    log("Training model on TRAIN split...")
    clf = model_mod.train_model(X_train, X_train["label"].to_numpy(), cfg.random_seed)

    log("Scoring VAL split and tuning threshold for macro F0.5...")
    val_scored = X_val[["s1_id", "cand_id"]].copy()
    val_scored["score"] = model_mod.predict_proba(clf, X_val)
    gt_sets = {k: set(v) for k, v in ground_truth.items()}

    if cfg.fixed_threshold is not None:
        best_threshold = cfg.fixed_threshold
        threshold_table = pd.DataFrame([{"threshold": best_threshold, "note": "fixed by user"}])
    else:
        best_threshold, threshold_table = evaluation.tune_threshold(
            val_scored, gt_sets, val_ids, cfg.thresholds
        )
    log(f"  best threshold = {best_threshold}")

    val_preds = evaluation.predictions_from_scores(val_scored, best_threshold)
    val_metrics = evaluation.macro_f_beta(val_preds, gt_sets, val_ids, beta=0.5)
    singleton_metrics = evaluation.singleton_performance(val_preds, gt_sets, val_ids)
    log(f"  VAL metrics: {val_metrics}")
    log(f"  VAL singleton metrics: {singleton_metrics}")

    # ---------------- Retrain on ALL labeled train data with tuned threshold ----------------
    log("Retraining final model on all labeled train+val pairs...")
    X_all = pd.concat([X_train, X_val], ignore_index=True)
    final_clf = model_mod.train_model(X_all, X_all["label"].to_numpy(), cfg.random_seed)

    model_mod.save_model(final_clf, transformers, cfg.models_dir / "model.joblib")

    # ---------------- Candidate generation (test) ----------------
    log("Generating TEST candidates (blocking)...")
    test_candidates = generate_candidates_all_countries(s1_test_n, s2_test_n, s3_test_n, cfg)
    n_pairs_test = sum(len(v) for v in test_candidates.values())
    log(f"  test candidate pairs: {n_pairs_test} across {len(test_candidates)} source1 entities")

    s1_test_order = s1_test["entity_id"].tolist()
    io_utils.write_candidate_pairs(cfg.output_dir / "candidate_pairs.tsv", test_candidates, s1_test_order)

    # ---------------- Score + predict test ----------------
    log("Scoring TEST candidates...")
    test_scored = predict.score_candidates(
        test_candidates, s1_test_n, cand_test_n, final_clf, transformers,
        n_jobs=cfg.n_jobs, chunk_size=cfg.feature_chunk_size,
    )
    test_matches = predict.apply_threshold(test_scored, best_threshold, s1_test_order)
    io_utils.write_matching_results(cfg.output_dir / "matching_results.tsv", test_matches, s1_test_order)

    # ---------------- Validate ----------------
    validate_outputs(
        cfg.output_dir / "matching_results.tsv", cfg.output_dir / "candidate_pairs.tsv", s1_test_order
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

    # ---------------- Report ----------------
    n_matched_test = sum(1 for v in test_matches.values() if v)
    report = {
        "best_threshold": best_threshold,
        "val_metrics": val_metrics,
        "val_singleton_metrics": singleton_metrics,
        "train_candidate_recall": rec,
        "train_candidate_reduction_ratio": reduction_ratio,
        "n_train_candidate_pairs": n_pairs_train,
        "n_test_candidate_pairs": n_pairs_test,
        "n_test_source1_entities": len(s1_test_order),
        "n_test_entities_with_a_match": n_matched_test,
        "validator_passed": validator_passed,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (cfg.reports_dir / "validation_metrics.json").write_text(json.dumps(report, indent=2, default=str))
    threshold_table.to_csv(cfg.reports_dir / "threshold_tuning.tsv", sep="\t", index=False)

    log("DONE.")
    log(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
