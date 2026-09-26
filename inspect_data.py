import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
import pandas as pd
import time

DATA_DIR = r"a:\Projects\Amazon ML\6ab10eb3b23ba_student_resource\student_resource\dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

def inspect_file(filepath, name):
    print("=" * 80)
    print(f"FILE: {name}")
    print(f"Path: {filepath}")
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")
    
    t0 = time.time()
    # Read first 5 rows to show preview quickly
    preview_df = pd.read_csv(filepath, sep="\t", nrows=5)
    print("\n--- First 3 Rows Preview ---")
    for idx, row in preview_df.head(3).iterrows():
        print(dict(row))
        
    print(f"\nColumns: {list(preview_df.columns)}")
    
    # Fast row counting and schema verification
    # Using chunksize to avoid memory spikes
    total_rows = 0
    null_counts = {col: 0 for col in preview_df.columns}
    country_counts = {}
    
    for chunk in pd.read_csv(filepath, sep="\t", chunksize=250000, low_memory=False):
        total_rows += len(chunk)
        for col in preview_df.columns:
            null_counts[col] += int(chunk[col].isna().sum())
        if "country" in chunk.columns:
            vc = chunk["country"].value_counts(dropna=False).to_dict()
            for k, v in vc.items():
                country_counts[k] = country_counts.get(k, 0) + v
                
    t1 = time.time()
    print(f"Total Rows: {total_rows:,} (scanned in {t1 - t0:.2f}s)")
    print("Missing values per column:")
    for col, n_null in null_counts.items():
        pct = (n_null / total_rows * 100) if total_rows > 0 else 0
        print(f"  - {col}: {n_null:,} ({pct:.2f}%)")
        
    if country_counts:
        print("Country distribution:")
        for c, count in country_counts.items():
            pct = count / total_rows * 100
            print(f"  - {c}: {count:,} ({pct:.2f}%)")

    return total_rows

def inspect_ground_truth(filepath):
    print("=" * 80)
    print(f"GROUND TRUTH ANALYSIS: {filepath}")
    t0 = time.time()
    df = pd.read_csv(filepath, sep="\t", dtype=str).fillna("")
    total_s1 = len(df)
    
    matched_ids_series = df["matched_entity_ids"]
    singletons = int((matched_ids_series == "").sum())
    non_singletons = total_s1 - singletons
    
    # Analyze match counts
    # For non-empty strings, count commas + 1
    non_empty = matched_ids_series[matched_ids_series != ""]
    match_lengths = pd.concat([
        pd.Series(0, index=df[matched_ids_series == ""].index),
        non_empty.str.count(",") + 1
    ])
    
    t1 = time.time()
    print(f"Total Source 1 Records in Ground Truth: {total_s1:,} (scanned in {t1 - t0:.2f}s)")
    print(f"Singletons (0 matches): {singletons:,} ({singletons / total_s1 * 100:.2f}%)")
    print(f"Records with >= 1 match: {non_singletons:,} ({non_singletons / total_s1 * 100:.2f}%)")
    print("\nMatch count statistics per Source 1 entity:")
    print(match_lengths.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
    print(f"Distribution of number of matches:")
    print(match_lengths.value_counts().sort_index().head(10))

if __name__ == "__main__":
    print("Starting Inspection of Amazon ML Datasets...")
    
    train_files = [
        ("train_source1.tsv", os.path.join(TRAIN_DIR, "train_source1.tsv")),
        ("train_source2.tsv", os.path.join(TRAIN_DIR, "train_source2.tsv")),
        ("train_source3.tsv", os.path.join(TRAIN_DIR, "train_source3.tsv")),
    ]
    
    test_files = [
        ("test_source1.tsv", os.path.join(TEST_DIR, "test_source1.tsv")),
        ("test_source2.tsv", os.path.join(TEST_DIR, "test_source2.tsv")),
        ("test_source3.tsv", os.path.join(TEST_DIR, "test_source3.tsv")),
    ]
    
    print("\n>>> INSPECTING TRAINING SOURCE FILES <<<")
    for name, path in train_files:
        inspect_file(path, name)
        
    print("\n>>> INSPECTING GROUND TRUTH <<<")
    inspect_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
    
    print("\n>>> INSPECTING TEST FILES <<<")
    for name, path in test_files:
        inspect_file(path, name)
