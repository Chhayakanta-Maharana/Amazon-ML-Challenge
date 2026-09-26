import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from preprocessing import clean_name, clean_address, clean_country
from blocking import Blocker
from features import extract_pairwise_features

def test_inference_speed():
    print("Testing inference speed on France partition slice...")
    # Load slice of test France
    df_s1 = pd.read_csv(Config.TEST_S1, sep="\t", nrows=5000)
    df_s2 = pd.read_csv(Config.TEST_S2, sep="\t", nrows=50000)
    df_s3 = pd.read_csv(Config.TEST_S3, sep="\t", nrows=50000)
    
    target_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    
    # Load Model
    model_data = joblib.load(os.path.join(Config.ARTIFACTS_DIR, "matching_model.joblib"))
    model = model_data["model"]
    threshold = model_data.get("threshold", 0.85)
    print(f"Loaded model with threshold {threshold:.2f}")
    
    # Preprocess & index targets
    t0 = time.time()
    blocker = Blocker(max_candidates_per_entity=40)
    blocker.index_target_records(target_pool)
    print(f"Indexed {len(target_pool)} targets in {time.time() - t0:.2f}s")
    
    target_dict = blocker.records
    
    # Preprocess S1 records and run inference
    t1 = time.time()
    total_candidates = 0
    total_matches = 0
    
    for row in df_s1.itertuples(index=False):
        cands, ch_hits = blocker.generate_candidates_for_entity(
            row.entity_id, row.business_name, row.business_address, row.country
        )
        total_candidates += len(cands)
        
        if cands:
            s1_country = clean_country(row.country)
            s1_name = clean_name(row.business_name)
            s1_addr = clean_address(row.business_address)
            
            cand_features = []
            valid_cands = []
            for cand in cands:
                if cand in target_dict:
                    t_c, t_n, t_a = target_dict[cand]
                    f = extract_pairwise_features(
                        s1_name, s1_addr, s1_country,
                        t_n, t_a, t_c,
                        channel_count=len(ch_hits[cand])
                    )
                    cand_features.append(f)
                    valid_cands.append(cand)
                    
            if cand_features:
                probs = model.predict_proba(np.array(cand_features))[:, 1]
                matches = [c for c, p in zip(valid_cands, probs) if p >= threshold]
                total_matches += len(matches)
                
    elapsed = time.time() - t1
    print(f"Processed {len(df_s1)} S1 entities in {elapsed:.2f}s ({len(df_s1)/elapsed:.1f} entities/s)")
    print(f"Total candidates: {total_candidates}, Matches: {total_matches}")

if __name__ == "__main__":
    test_inference_speed()
