"""Central configuration for the business entity resolution pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    train_dir: Path = Path("dataset/train")
    test_dir: Path = Path("dataset/test")
    output_dir: Path = Path("output")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    random_seed: int = 42
    val_fraction: float = 0.2

    # Candidate generation
    n_neighbors: int = 15          # TF-IDF nearest-neighbor blocking, per side
    max_candidates: int = 50       # cap on candidates kept per source1 entity (post-union, pre-model)
    prefix_len: int = 4            # prefix-blocking key length
    min_token_len_for_index: int = 4  # min token length used for inverted-index blocking
    n_jobs: int = -1               # parallel workers for blocking/features; -1 = all cores
    feature_chunk_size: int = 200_000  # pairs per chunk when parallelizing feature computation

    # Thresholds to try when tuning for macro F0.5
    thresholds: List[float] = field(
        default_factory=lambda: [round(0.05 * i, 2) for i in range(4, 20)]
    )
    # Optional fixed threshold override (skips tuning if set, e.g. via --prediction-threshold)
    fixed_threshold: Optional[float] = None

    # Memory-safe per-country streaming
    phase: str = "all"                        # "train", "predict", or "all"
    partitioned_train_dir: Optional[Path] = None  # default: <train_dir>_by_country
    partitioned_test_dir: Optional[Path] = None   # default: <test_dir>_by_country
    force_repartition: bool = False

    def __post_init__(self):
        self.train_dir = Path(self.train_dir)
        self.test_dir = Path(self.test_dir)
        self.output_dir = Path(self.output_dir)
        self.models_dir = Path(self.models_dir)
        self.reports_dir = Path(self.reports_dir)
        if self.partitioned_train_dir is None:
            self.partitioned_train_dir = self.train_dir.parent / f"{self.train_dir.name}_by_country"
        else:
            self.partitioned_train_dir = Path(self.partitioned_train_dir)
        if self.partitioned_test_dir is None:
            self.partitioned_test_dir = self.test_dir.parent / f"{self.test_dir.name}_by_country"
        else:
            self.partitioned_test_dir = Path(self.partitioned_test_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Business entity resolution pipeline")
    p.add_argument("--train-dir", type=str, default="dataset/train")
    p.add_argument("--test-dir", type=str, default="dataset/test")
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--max-candidates", type=int, default=50)
    p.add_argument("--prediction-threshold", type=float, default=None,
                    help="If set, skip threshold tuning and use this fixed threshold.")
    p.add_argument("--n-jobs", type=int, default=-1,
                    help="Parallel workers for blocking/feature computation. -1 = all cores.")
    p.add_argument("--feature-chunk-size", type=int, default=200_000,
                    help="Pairs per chunk when parallelizing feature computation.")
    p.add_argument("--phase", type=str, choices=["train", "predict", "all"], default="all",
                    help="Run only training, only prediction (loads a saved model bundle), or both.")
    p.add_argument("--partitioned-train-dir", type=str, default=None,
                    help="Where to write/read per-country train partitions. Default: <train-dir>_by_country")
    p.add_argument("--partitioned-test-dir", type=str, default=None,
                    help="Where to write/read per-country test partitions. Default: <test-dir>_by_country")
    p.add_argument("--force-repartition", action="store_true",
                    help="Delete and rebuild the per-country partition cache even if it already exists.")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        random_seed=args.random_seed,
        val_fraction=args.val_fraction,
        n_neighbors=args.n_neighbors,
        max_candidates=args.max_candidates,
        partitioned_train_dir=args.partitioned_train_dir,
        partitioned_test_dir=args.partitioned_test_dir,
    )
    cfg.fixed_threshold = args.prediction_threshold
    cfg.n_jobs = args.n_jobs
    cfg.feature_chunk_size = args.feature_chunk_size
    cfg.phase = args.phase
    cfg.force_repartition = args.force_repartition
    return cfg