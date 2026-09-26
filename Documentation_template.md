# Methodology Document — Business Entity Resolution

## 1. Methodology

A two-stage entity-resolution pipeline: (1) high-recall **candidate
generation / blocking** to cut the O(|S1|·(|S2|+|S3|)) search space down to a
tractable set of plausible pairs, then (2) a **supervised binary classifier**
over pairwise similarity features to decide which candidates are true
matches, with the decision threshold tuned specifically for macro F0.5.

## 2. Candidate generation / blocking

Implemented in `src/blocking.py`, run separately for S1×S2 and S1×S3, and
**country-partitioned** first — verified against 172k+ ground-truth matched
pairs to have zero cross-country matches, so this loses no true positives
while cutting the per-partition problem size by roughly the number of
distinct countries. Within each country partition, six rules are unioned:
exact normalized-name match, name-prefix blocks, country+first-token blocks,
shared numeric/postal tokens, bucketed character-n-gram TF-IDF nearest
neighbors, and address-token overlap. All are implemented as vectorized
pandas merges (no per-row Python loops), and overly common keys
(> `MAX_BLOCK_SIZE` records) are dropped before joining to bound pair
explosion. Per-entity candidate counts are then capped (`--max-candidates`,
default 50) by a cheap token-Jaccard ranking.

## 3. Feature engineering

`src/features.py` computes ~28 pairwise features spanning exact/fuzzy name
similarity (RapidFuzz ratio/WRatio/token-set/token-sort, normalized
Levenshtein, token Jaccard/containment, char n-gram TF-IDF cosine, a
rarity-weighted shared-token score), address similarity (exact/edit/overlap/
Jaccard/TF-IDF cosine, shared numeric tokens, postal and street-number
agreement), and cross-cutting signals (country match, source-pair indicator,
missing-value flags, combined and disagreement scores). All TF-IDF
vectorizers and the word-IDF rarity table are **fit once on training data**
and reused unchanged (`.transform` only) for validation and test.

## 4. Model architecture

`sklearn.ensemble.HistGradientBoostingClassifier`, chosen for its permissive
BSD license, native handling of mixed-scale numeric features without manual
preprocessing, and practical training speed on large pair counts. Trained
with class-balanced sample weights, since true matches are a small minority
of blocked candidate pairs.

## 5. Training and validation

Ground truth is converted into pairwise labels: every candidate pair is `1`
if it's a true match, `0` otherwise, which naturally produces hard negatives
(high-similarity pairs that blocked together with the true match but aren't
it). The split is performed **by Source 1 entity id**, not by individual
pair, with a fixed random seed, so no entity's pairs leak across the split.
After threshold tuning on the validation split, a final model is retrained
on the combined train+validation labels before scoring test.

## 6. Threshold selection

Thresholds in a configurable list are evaluated on the validation split; the
one maximizing **macro-averaged F0.5 over Source 1 entities** is selected.
Because F0.5 weights precision twice as heavily as recall, the objective
itself naturally favors conservative thresholds — the pipeline additionally
never forces a match (empty match lists are valid and expected) and applies
the threshold independently per candidate pair, so an entity can end up with
zero, one, or several matches.

## 7. Limitations

- The bucketed TF-IDF nearest-neighbor blocking rule skips buckets above
  `TFIDF_BUCKET_MAX`×`TFIDF_BUCKET_MAX` pair combinations for tractability;
  extremely common name prefixes rely on the other five blocking rules for
  recall in that case.
- Non-Latin-script names (e.g. Devanagari) are transliterated with
  `unidecode`, a generic phonetic/lookup transliteration — this is a
  reasonable approximation, not a linguistically precise romanization, and
  will be noisier than same-script comparisons.
- Candidate capping (`--max-candidates`) trades a small amount of recall for
  bounded compute at very large scale; raising it improves recall at the
  cost of runtime.
- Country-partitioned blocking assumes matches never cross countries; this
  was verified empirically on the provided training data but is not a
  logical guarantee for arbitrary future data.

## 8. Reproducibility

All randomness (train/val split, model initialization) is controlled by
`--random-seed` (default 42). Given identical input files and the same seed,
the pipeline is deterministic end-to-end, including the final output file
row ordering (source1 entities appear in their original test-file order;
matched/candidate ids within a row are sorted deterministically).

## 9. License compliance

All libraries used (pandas, numpy, scikit-learn, rapidfuzz, scipy, joblib,
Unidecode) are permissively licensed (BSD/MIT/Apache-family). No GPL or
similarly restrictive dependencies are introduced.

## 10. Confirmation: no external data lookup

No network calls, geocoding services, business-database lookups, or
third-party APIs are used anywhere in this pipeline. `normalization.py`'s
transliteration step (`unidecode`) is a local, offline lookup table shipped
with the package — it performs no network I/O. All matching decisions are
derived solely from the `business_name`, `business_address`, and `country`
fields present in the input TSVs.
