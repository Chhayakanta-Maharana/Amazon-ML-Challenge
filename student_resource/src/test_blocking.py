import os
import sys
import time
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from blocking import Blocker
from config import Config

def test_blocking_recall():
    print("="*60)
    print("PHASE 10 & 11: CANDIDATE GENERATION & RECALL ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Load subset of training data
    print("Loading test slice of training data...")
    n_s1 = 20000
    n_s23 = 150000
    
    s1 = pd.read_csv(Config.TRAIN_S1, sep="\t", nrows=n_s1)
    s2 = pd.read_csv(Config.TRAIN_S2, sep="\t", nrows=n_s23)
    s3 = pd.read_csv(Config.TRAIN_S3, sep="\t", nrows=n_s23)
    gt = pd.read_csv(Config.TRAIN_GT, sep="\t", nrows=n_s1)
    
    # Combine S2 and S3 for target candidate pool
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"Loaded {len(s1)} S1 entities, {len(s23)} target records (S2+S3)")
    
    # Build Blocker index
    blocker = Blocker(max_candidates_per_entity=50)
    print("Building inverted indices...")
    idx_start = time.time()
    blocker.index_target_records(s23)
    print(f"Indexing completed in {time.time() - idx_start:.2f}s")
    
    # Evaluate recall on GT pairs where true target ID is in s23
    s23_valid_ids = set(s23['entity_id'])
    
    total_true_pairs_in_pool = 0
    recovered_true_pairs = 0
    candidate_counts = []
    
    eval_start = time.time()
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = str(row['matched_entity_ids'])
        
        s1_row = s1[s1['entity_id'] == s1_id]
        if s1_row.empty:
            continue
        s1_rec = s1_row.iloc[0]
        
        true_matches = [m.strip() for m in matches_str.split(',') if m.strip() and m.strip() in s23_valid_ids]
        
        cands, channel_hits = blocker.generate_candidates_for_entity(
            s1_id, s1_rec['business_name'], s1_rec['business_address'], s1_rec['country']
        )
        cand_set = set(cands)
        candidate_counts.append(len(cands))
        
        for tm in true_matches:
            total_true_pairs_in_pool += 1
            if tm in cand_set:
                recovered_true_pairs += 1
                
    eval_time = time.time() - eval_start
    recall = (recovered_true_pairs / max(1, total_true_pairs_in_pool)) * 100
    
    print("\n--- Blocking Evaluation Results ---")
    print(f"Total True Pairs Evaluated (present in pool): {total_true_pairs_in_pool}")
    print(f"Recovered True Pairs: {recovered_true_pairs}")
    print(f"Candidate Generation Recall: {recall:.2f}%")
    print(f"Average Candidates per S1: {np.mean(candidate_counts):.2f}")
    print(f"Median Candidates per S1: {np.median(candidate_counts):.1f}")
    print(f"95th Percentile Candidates: {np.percentile(candidate_counts, 95):.1f}")
    print(f"Max Candidates: {max(candidate_counts) if candidate_counts else 0}")
    print(f"Candidate Generation Throughput: {len(s1) / eval_time:.1f} entities/sec")
    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_blocking_recall()
