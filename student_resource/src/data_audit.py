import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

def audit_dataset(base_dir="student_resource/dataset"):
    print("="*60)
    print("PHASE 2 & 3: DATA LOADING & DEEP DATA QUALITY AUDIT")
    print("="*60)
    
    files = {
        "train_s1": os.path.join(base_dir, "train", "train_source1.tsv"),
        "train_s2": os.path.join(base_dir, "train", "train_source2.tsv"),
        "train_s3": os.path.join(base_dir, "train", "train_source3.tsv"),
        "train_gt": os.path.join(base_dir, "train", "train_ground_truth.tsv"),
        "test_s1": os.path.join(base_dir, "test", "test_source1.tsv"),
        "test_s2": os.path.join(base_dir, "test", "test_source2.tsv"),
        "test_s3": os.path.join(base_dir, "test", "test_source3.tsv"),
    }
    
    # 1. Check file existence & sizes
    for k, path in files.items():
        if not os.path.isfile(path):
            print(f"[ERROR] Missing file: {path}")
            return
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"File: {k:8s} | Path: {path} | Size: {size_mb:.2f} MB")
        
    print("\n--- Inspecting Train Files ---")
    
    # Read Train Source 1
    print("\nLoading train_source1.tsv...")
    df_train_s1 = pd.read_csv(files["train_s1"], sep="\t", dtype=str)
    print(f"train_s1 shape: {df_train_s1.shape}")
    print(f"train_s1 columns: {list(df_train_s1.columns)}")
    print(f"train_s1 missing: {df_train_s1.isnull().sum().to_dict()}")
    print(f"train_s1 unique entity_ids: {df_train_s1['entity_id'].nunique()}")
    print(f"train_s1 countries: {df_train_s1['country'].value_counts(dropna=False).to_dict()}")
    s1_prefixes = df_train_s1['entity_id'].str[:3].value_counts().to_dict()
    print(f"train_s1 ID prefixes: {s1_prefixes}")
    
    # Read Train Source 2
    print("\nLoading train_source2.tsv...")
    df_train_s2 = pd.read_csv(files["train_s2"], sep="\t", dtype=str)
    print(f"train_s2 shape: {df_train_s2.shape}")
    print(f"train_s2 missing: {df_train_s2.isnull().sum().to_dict()}")
    print(f"train_s2 unique entity_ids: {df_train_s2['entity_id'].nunique()}")
    print(f"train_s2 countries: {df_train_s2['country'].value_counts(dropna=False).to_dict()}")
    s2_prefixes = df_train_s2['entity_id'].str[:3].value_counts().to_dict()
    print(f"train_s2 ID prefixes: {s2_prefixes}")

    # Read Train Source 3
    print("\nLoading train_source3.tsv...")
    df_train_s3 = pd.read_csv(files["train_s3"], sep="\t", dtype=str)
    print(f"train_s3 shape: {df_train_s3.shape}")
    print(f"train_s3 missing: {df_train_s3.isnull().sum().to_dict()}")
    print(f"train_s3 unique entity_ids: {df_train_s3['entity_id'].nunique()}")
    print(f"train_s3 countries: {df_train_s3['country'].value_counts(dropna=False).to_dict()}")
    s3_prefixes = df_train_s3['entity_id'].str[:3].value_counts().to_dict()
    print(f"train_s3 ID prefixes: {s3_prefixes}")

    # Read Train Ground Truth
    print("\nLoading train_ground_truth.tsv...")
    df_train_gt = pd.read_csv(files["train_gt"], sep="\t", dtype=str)
    print(f"train_gt shape: {df_train_gt.shape}")
    print(f"train_gt columns: {list(df_train_gt.columns)}")
    print(f"train_gt missing: {df_train_gt.isnull().sum().to_dict()}")
    print(f"train_gt unique source1_entity_ids: {df_train_gt['source1_entity_id'].nunique()}")
    
    # Audit Ground Truth Matches
    print("\nAuditing Ground Truth Match Structure...")
    # Fill NaN matched_entity_ids with empty string
    matched_series = df_train_gt['matched_entity_ids'].fillna("")
    
    match_counts = []
    s2_match_counts = []
    s3_match_counts = []
    s2_valid_ids = set(df_train_s2['entity_id'])
    s3_valid_ids = set(df_train_s3['entity_id'])
    s1_valid_ids = set(df_train_s1['entity_id'])
    
    invalid_match_ids = []
    
    for _, row in df_train_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ""
        if not matches_str or matches_str.strip() == "":
            match_counts.append(0)
            s2_match_counts.append(0)
            s3_match_counts.append(0)
        else:
            ids = [x.strip() for x in matches_str.split(",") if x.strip()]
            match_counts.append(len(ids))
            s2_c = sum(1 for x in ids if x.startswith("S2-"))
            s3_c = sum(1 for x in ids if x.startswith("S3-"))
            s2_match_counts.append(s2_c)
            s3_match_counts.append(s3_c)
            
            # Sample check for invalid IDs
            if len(invalid_match_ids) < 10:
                for x in ids:
                    if x.startswith("S2-") and x not in s2_valid_ids:
                        invalid_match_ids.append((s1_id, x, "Missing from S2"))
                    elif x.startswith("S3-") and x not in s3_valid_ids:
                        invalid_match_ids.append((s1_id, x, "Missing from S3"))
                    elif not (x.startswith("S2-") or x.startswith("S3-")):
                        invalid_match_ids.append((s1_id, x, "Invalid prefix"))
                        
    mc_counter = Counter(match_counts)
    print(f"Total Source 1 in GT: {len(df_train_gt)}")
    print(f"Match count distribution (matches per S1):")
    for k in sorted(mc_counter.keys())[:15]:
        pct = mc_counter[k] / len(df_train_gt) * 100
        print(f"  {k} matches: {mc_counter[k]} entities ({pct:.2f}%)")
        
    print(f"Singletons (0 matches): {mc_counter[0]} ({mc_counter[0]/len(df_train_gt)*100:.2f}%)")
    print(f"Single match (1 match): {mc_counter[1]} ({mc_counter[1]/len(df_train_gt)*100:.2f}%)")
    print(f"Multi-match (>=2 matches): {sum(v for k,v in mc_counter.items() if k >= 2)} ({sum(v for k,v in mc_counter.items() if k >= 2)/len(df_train_gt)*100:.2f}%)")
    print(f"Max matches for a single S1 entity: {max(mc_counter.keys())}")
    print(f"Invalid match ID sample (should be empty): {invalid_match_ids}")
    
    # Check Test Files
    print("\n--- Inspecting Test Files ---")
    print("\nLoading test_source1.tsv...")
    df_test_s1 = pd.read_csv(files["test_s1"], sep="\t", dtype=str)
    print(f"test_s1 shape: {df_test_s1.shape}")
    print(f"test_s1 missing: {df_test_s1.isnull().sum().to_dict()}")
    print(f"test_s1 unique entity_ids: {df_test_s1['entity_id'].nunique()}")
    print(f"test_s1 countries: {df_test_s1['country'].value_counts(dropna=False).to_dict()}")
    
    print("\nLoading test_source2.tsv...")
    df_test_s2 = pd.read_csv(files["test_s2"], sep="\t", dtype=str)
    print(f"test_s2 shape: {df_test_s2.shape}")
    print(f"test_s2 missing: {df_test_s2.isnull().sum().to_dict()}")
    print(f"test_s2 unique entity_ids: {df_test_s2['entity_id'].nunique()}")
    print(f"test_s2 countries: {df_test_s2['country'].value_counts(dropna=False).to_dict()}")

    print("\nLoading test_source3.tsv...")
    df_test_s3 = pd.read_csv(files["test_s3"], sep="\t", dtype=str)
    print(f"test_s3 shape: {df_test_s3.shape}")
    print(f"test_s3 missing: {df_test_s3.isnull().sum().to_dict()}")
    print(f"test_s3 unique entity_ids: {df_test_s3['entity_id'].nunique()}")
    print(f"test_s3 countries: {df_test_s3['country'].value_counts(dropna=False).to_dict()}")

    print("\n" + "="*60)
    print("AUDIT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    audit_dataset()
