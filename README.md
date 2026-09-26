# Business Entity Resolution

Match noisy business records from **Source 2** and **Source 3** against a
deduplicated reference set, **Source 1**, using a two-stage
blocking-then-classification entity-resolution pipeline, optimized for
**macro F0.5**.

## 1. Project purpose

Given three record sources sharing the schema
`entity_id, business_name, business_address, country`, decide — for every
Source 1 entity — which Source 2 and Source 3 records refer to the same
real-world business. No external data, geocoding, or APIs are used anywhere;
everything runs on the text in the files themselves.

## 2. Dataset assumptions

- Inputs are **tab-separated** files loaded with `pd.read_csv(path, sep="\t")`.
- `entity_id` prefixes (`S1-`, `S2-`, `S3-`) identify the source.
- `train_ground_truth.tsv` has `source1_entity_id, matched_entity_ids`
  (comma-separated Source 2/3 ids; may be empty).
- **Scale**: the real dataset is large — millions of rows per source (this
  is not a toy hackathon-sized file). The pipeline is built accordingly
  (see §7 and the Performance notes below), not as a naive O(n·m) join.
- Test data may contain countries absent from training (verified: training
  only has `US`/`India`, test additionally has `France`). Nothing in the
  code hard-codes a country list — see §7.
- Empirically, on the training ground truth, **100% of true matches share
  the same `country` value** as their Source 1 record (checked directly on
  172k+ matched pairs). Country-partitioning is used as a blocking
  optimization for this reason — it does not filter or exclude any country,
  every country present in the data is processed.
- Source 1 names are always Latin-script/English; Source 2/Source 3 sometimes
  contain vernacular scripts (e.g. Devanagari) or accented Latin (French).
  See §6 on transliteration.

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Exact commands

```bash
# Full pipeline: train, tune, predict on test, validate
python -m src.run_pipeline \
  --train-dir dataset/train \
  --test-dir dataset/test \
  --output-dir output

# Faster dev/smoke-test loop: pin a threshold, fewer parallel workers, etc.
python -m src.run_pipeline \
  --train-dir dataset/train --test-dir dataset/test --output-dir output \
  --n-jobs 4 --max-candidates 30
```

All arguments (see `src/config.py` / `--help`):
`--train-dir --test-dir --output-dir --models-dir --reports-dir
--random-seed --val-fraction --n-neighbors --max-candidates
--prediction-threshold --n-jobs --feature-chunk-size`.

### Smoke-testing on a subset (recommended before a full run)

Because the full dataset is large, **first run the pipeline on a small
subsample** to catch bugs quickly, e.g. by pointing `--train-dir`/`--test-dir`
at copies of the TSVs truncated with `head`. A subsample must stay
*referentially consistent*: every id referenced in `matched_entity_ids`
in your truncated ground truth should exist in your truncated
source2/source3 files, or candidate recall will look artificially low.

## 5. Pipeline architecture

```
Input TSVs
  -> io_utils        (load + schema validation)
  -> normalization    (name/address cleanup, per-record)
  -> blocking          (candidate generation, per (s1_id) -> [cand_ids])
  -> features           (pairwise similarity features, train-fitted transformers)
  -> model               (HistGradientBoostingClassifier, trained on labeled pairs)
  -> evaluation            (entity-level split, macro-F0.5 threshold tuning)
  -> predict                (score + threshold test candidates)
  -> io_utils                (write matching_results.tsv / candidate_pairs.tsv)
```
Orchestrated end-to-end by `src/run_pipeline.py`.

## 6. Normalization approach (`src/normalization.py`)

- Lowercasing, then **transliteration** via the local `unidecode` library
  (pure lookup-table based, no network calls) — this both strips accents
  from French text (`École` → `Ecole`) and gives a best-effort phonetic
  Latin rendering of non-Latin scripts (Devanagari, etc.), which matters
  because Source 1 names are always Latin-script while Source 2/3 sometimes
  are not. Falls back to plain ASCII-folding if `unidecode` isn't installed.
- `&` → `and`, punctuation stripped, whitespace collapsed.
- Legal-suffix normalization for names (`corp`↔`corporation`, `pvt`↔`private`,
  `ltd`↔`limited`, `inc`↔`incorporated`, plus `co`, `llc`, `llp`, `plc`).
- Address abbreviation normalization (`rd`↔`road`, `st`↔`street`,
  `ave`↔`avenue`, `apt`↔`apartment`, plus `blvd`, `dr`, `ln`, `hwy`, `fl`,
  `bldg`, `ste`, `sq`, `pl`, `ct`, `no`).
- Numbers (street numbers, postal codes, unit numbers) are **preserved**,
  never stripped as punctuation.
- `postal_code_candidates`: numeric tokens of length ≥ 4 (works across
  5-digit US ZIPs, 6-digit Indian PIN codes, and other formats without
  hard-coding a country-specific pattern).
- `street_number_candidate`: first purely-numeric token within the first
  6 words of the normalized address.

## 7. Blocking strategy (`src/blocking.py`, `generate_candidates_all_countries`)

Run **separately** for Source1×Source2 and Source1×Source3, then unioned.
Candidate generation is **country-partitioned** first (a safe, verified
optimization — not a filter, see §2), and every rule below is a vectorized
pandas merge/join, not a per-row Python loop:

1. Exact normalized-name match.
2. Name-prefix blocking (first 4 normalized characters).
3. Country + first-name-token blocking.
4. Shared postal-code-length numeric token, and shared numeric token
   generally (from both name and address).
5. Character n-gram (2–4) TF-IDF nearest-neighbor blocking — computed
   **bucket-by-bucket** on (country, name-prefix) sub-buckets so it stays
   tractable at multi-million-row scale; oversized buckets are skipped
   (other rules cover their recall).
6. Address-token overlap (tokens of length ≥ 4, excluding pure numbers).

Keys shared by more than `MAX_BLOCK_SIZE` (400) records on either side are
dropped as non-discriminative before joining, to keep pair counts bounded.
After the union, each Source 1 entity's candidates are capped at
`--max-candidates` (default 50), ranked by a cheap token-Jaccard score —
**recall-preserving is the goal**, the cap only trims the long tail.
The exact candidate set fed to the model (for the **test** set) is written
to `output/candidate_pairs.tsv`.

## 8. Feature engineering (`src/features.py`)

~28 features per candidate pair — see `FEATURE_COLUMNS` for the full list:
RapidFuzz ratio/WRatio/token-set/token-sort, normalized Levenshtein
similarity, token Jaccard/containment, char n-gram TF-IDF cosine (both name
and address, via a fitted `TfidfVectorizer`), a rarity-weighted shared-token
score (word-level IDF), shared numeric tokens, postal-code and street-number
agreement, length differences, missing-value indicators, country match,
source-pair indicator, and combined/disagreement scores.

**TF-IDF vectorizers and the rare-token IDF table are fit once, on training
data only** (`fit_feature_transformers`), and reused unchanged (`.transform`
only) for validation and test — never re-fit on test data.

Feature computation is parallelized across CPU cores by chunking the pairs
(`compute_features_parallel`), since at full scale there can be tens of
millions of candidate pairs.

## 9. Model choice (`src/model.py`)

`sklearn.ensemble.HistGradientBoostingClassifier` (BSD-licensed): handles
mixed-scale numeric features without preprocessing, trains fast even on
large pair counts, and has built-in early stopping. Trained with
class-balanced sample weights (positives are a small minority of candidate
pairs).

## 10. Threshold tuning (`src/evaluation.py`)

The pipeline evaluates a configurable list of thresholds (default
`0.20 … 0.95` step `0.05`) against the **validation** split and picks the one
maximizing **macro F0.5** (averaged per Source 1 entity, not per pair).
Conservative thresholds are favored implicitly by the F0.5 objective itself
(precision weighted 2× recall). Empty match lists are allowed and expected
for entities with no true match (~5.6% of the real training ground truth).
`--prediction-threshold` can pin a fixed value and skip tuning.

## 11. Evaluation methodology

- Split by **Source 1 entity id** (`evaluation.group_split`), not by
  individual pair, so no entity leaks across train/validation.
- Reports: macro F0.5, micro precision/recall/F0.5, singleton-entity macro
  F0.5 (entities with exactly one true match), candidate recall (fraction of
  true matches present in the candidate set), and candidate reduction ratio
  (fraction of the full S1×(S2∪S3) cross product eliminated by blocking).
- After tuning, the final model is **retrained on train+val combined** with
  the tuned threshold fixed, then applied to test.

## 12. Output format

- `output/candidate_pairs.tsv`: `source1_entity_id, candidate_entity_ids`
  (comma-separated, no duplicates), one row per **test** Source 1 entity.
- `output/matching_results.tsv`: `source1_entity_id, matched_entity_ids`,
  one row per test Source 1 entity, empty string for no-match entities,
  matched ids always a subset of that entity's row in `candidate_pairs.tsv`.
- `run_pipeline.validate_outputs` checks these invariants automatically
  before the run is considered complete; `utils/validate_submission.py` is
  also invoked automatically if present.

## 13. Reproducing final results

Fixed `--random-seed` (default 42) controls the train/val split and the
model's internal randomness. Given the same input files and the same seed,
`python -m src.run_pipeline ...` reproduces the same candidate sets, the
same trained model, the same tuned threshold, and the same output files.

## Performance notes (please read before a full run)

The real dataset has **millions of rows per source**. This pipeline is
engineered for that scale (vectorized pandas joins for blocking,
bucket-limited TF-IDF NN, multiprocess-parallel feature computation via
`--n-jobs`), but a full run is still a genuinely heavy computation and can
take a long time depending on your machine's core count and RAM. Recommended
workflow:
1. Smoke-test on a small, referentially-consistent subsample first.
2. Run the full pipeline with `--n-jobs` set to your CPU core count.
3. Watch `reports/validation_metrics.json` / stdout logs for candidate
   counts per country — an unexpectedly huge count for one country usually
   means `MAX_BLOCK_SIZE` in `src/blocking.py` needs tightening for that
   partition.
