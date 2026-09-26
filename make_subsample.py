"""Build a small, referentially-consistent train/test subsample for smoke-testing.
Run from the project root: python make_subsample.py
"""
import pandas as pd
from pathlib import Path

TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")
OUT_TRAIN = Path("dataset/train_small")
OUT_TEST = Path("dataset/test_small")
N_S1 = 3000
CHUNK = 200_000

OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_TEST.mkdir(parents=True, exist_ok=True)

print("Reading source1 sample...")
s1 = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=N_S1)
s1_ids = set(s1["entity_id"])
s1.to_csv(OUT_TRAIN / "train_source1.tsv", sep="\t", index=False)
print(f"  {len(s1)} rows")

print("Scanning ground truth for matching rows (chunked)...")
gt_chunks = []
for chunk in pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str,
                          keep_default_na=False, chunksize=CHUNK):
    gt_chunks.append(chunk[chunk["source1_entity_id"].isin(s1_ids)])
gt = pd.concat(gt_chunks, ignore_index=True)
print(f"  {len(gt)} ground-truth rows found")

ref_ids = set()
for raw in gt["matched_entity_ids"]:
    if raw:
        ref_ids.update(raw.split(","))
ref_s2 = {x for x in ref_ids if x.startswith("S2-")}
ref_s3 = {x for x in ref_ids if x.startswith("S3-")}
print(f"  referenced: {len(ref_s2)} source2 ids, {len(ref_s3)} source3 ids")

def build_side(path, ref_ids, extra_n, out_path):
    matched_chunks = []
    extra = None
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK):
        matched_chunks.append(chunk[chunk["entity_id"].isin(ref_ids)])
        if extra is None:
            extra = chunk.head(extra_n)
    out = pd.concat(matched_chunks + [extra], ignore_index=True).drop_duplicates("entity_id")
    out.to_csv(out_path, sep="\t", index=False)
    print(f"  {path.name}: {len(out)} rows written")
    return set(out["entity_id"])

print("Building source2 sample...")
s2_ids = build_side(TRAIN_DIR / "train_source2.tsv", ref_s2, 4000, OUT_TRAIN / "train_source2.tsv")
print("Building source3 sample...")
s3_ids = build_side(TRAIN_DIR / "train_source3.tsv", ref_s3, 4000, OUT_TRAIN / "train_source3.tsv")

print("Filtering ground truth to only ids present in the sample...")
def filt(raw):
    if not raw:
        return ""
    return ",".join(x for x in raw.split(",") if x in s2_ids or x in s3_ids)
gt["matched_entity_ids"] = gt["matched_entity_ids"].apply(filt)
gt.to_csv(OUT_TRAIN / "train_ground_truth.tsv", sep="\t", index=False)
print(f"  {(gt['matched_entity_ids'] != '').sum()} non-empty rows after filtering")

print("Building test subsample (naive truncation, for format-checking only)...")
pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=2000)\
    .to_csv(OUT_TEST / "test_source1.tsv", sep="\t", index=False)
pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=4000)\
    .to_csv(OUT_TEST / "test_source2.tsv", sep="\t", index=False)
pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=4000)\
    .to_csv(OUT_TEST / "test_source3.tsv", sep="\t", index=False)

print("Done.")