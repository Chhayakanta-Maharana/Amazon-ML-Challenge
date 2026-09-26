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

def run_test_inference(sample_size=None):
    """
    Runs end-to-end test inference to generate matching_results.tsv and candidate_pairs.tsv.
    Supports country-by-country partitioned streaming to minimize RAM and maximize execution speed.
    """
    print("=" * 70)
    print("RUNNING END-TO-END INFERENCE ON TEST DATASET")
    print("=" * 70)
    start_total = time.time()
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model
    model_path = os.path.join(Config.ARTIFACTS_DIR, "matching_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run train_pipeline.py first.")
        
    print(f"Loading trained matching model from: {model_path}")
    model_artifact = joblib.load(model_path)
    model = model_artifact["model"]
    threshold = model_artifact.get("threshold", 0.85)
    print(f"Model loaded. Decision Threshold: {threshold:.2f}")
    
    # 2. Inspect Test S1 Records
    print(f"\nReading Test Source 1 records from: {Config.TEST_S1}")
    df_s1 = pd.read_csv(Config.TEST_S1, sep="\t", nrows=sample_size)
    print(f"Total Source 1 test entities to process: {len(df_s1):,}")
    
    # Initialize output file writers
    matching_out_path = Config.MATCHING_RESULTS
    candidate_out_path = Config.CANDIDATE_PAIRS
    
    print(f"\nPreparing Output Files:\n  - {matching_out_path}\n  - {candidate_out_path}")
    
    # Dicts to store results per S1 ID in exact original order
    final_matches_map = {}
    final_candidates_map = {}
    
    # Group S1 by cleaned country
    df_s1['clean_country'] = df_s1['country'].apply(clean_country)
    country_groups = df_s1.groupby('clean_country')
    
    # 3. Process partition by partition
    for country, s1_group in country_groups:
        print(f"\n" + "-" * 50)
        print(f"Processing Country Partition: [{country}] ({len(s1_group):,} S1 Entities)")
        print("-" * 50)
        c_start = time.time()
        
        # Load S2 and S3 for this country in chunks
        print(f"Loading Test S2 and S3 records for country: {country}...")
        s2_chunks = []
        for chunk in pd.read_csv(Config.TEST_S2, sep="\t", chunksize=250000):
            chunk_filtered = chunk[chunk['country'].apply(clean_country) == country]
            if len(chunk_filtered) > 0:
                s2_chunks.append(chunk_filtered)
                
        s3_chunks = []
        for chunk in pd.read_csv(Config.TEST_S3, sep="\t", chunksize=250000):
            chunk_filtered = chunk[chunk['country'].apply(clean_country) == country]
            if len(chunk_filtered) > 0:
                s3_chunks.append(chunk_filtered)
                
        target_records = pd.concat(s2_chunks + s3_chunks, ignore_index=True)
        print(f"Loaded {len(target_records):,} target records (S2 + S3) for [{country}].")
        
        # Build Blocker index for this country
        print("Building inverted candidate index...")
        idx_t0 = time.time()
        blocker = Blocker(max_candidates_per_entity=Config.MAX_CANDIDATES_PER_S1)
        blocker.index_target_records(target_records)
        print(f"Candidate indexing completed in {time.time() - idx_t0:.2f}s.")
        
        target_dict = blocker.records
        
        # Run S1 candidate generation & scoring
        print(f"Running candidate generation & model scoring on {len(s1_group):,} S1 records...")
        inf_t0 = time.time()
        total_cands = 0
        total_matches = 0
        
        batch_records = s1_group[['entity_id', 'business_name', 'business_address', 'country']].to_dict(orient='records')
        
        for i, s1_rec in enumerate(batch_records):
            s1_id = s1_rec['entity_id']
            cands, ch_hits = blocker.generate_candidates_for_entity(
                s1_id, s1_rec['business_name'], s1_rec['business_address'], country
            )
            
            final_candidates_map[s1_id] = cands
            total_cands += len(cands)
            
            matched_ids = []
            if cands:
                s1_n = clean_name(s1_rec['business_name'])
                s1_a = clean_address(s1_rec['business_address'])
                
                cand_features = []
                valid_cands = []
                for cand in cands:
                    if cand in target_dict:
                        t_c, t_n, t_a = target_dict[cand]
                        f = extract_pairwise_features(
                            s1_n, s1_a, country,
                            t_n, t_a, t_c,
                            channel_count=len(ch_hits[cand])
                        )
                        cand_features.append(f)
                        valid_cands.append(cand)
                        
                if cand_features:
                    probs = model.predict_proba(np.array(cand_features))[:, 1]
                    matched_ids = [c for c, p in zip(valid_cands, probs) if p >= threshold]
                    total_matches += len(matched_ids)
                    
            final_matches_map[s1_id] = matched_ids
            
            if (i + 1) % 25000 == 0 or (i + 1) == len(batch_records):
                print(f"  Processed {i + 1:,}/{len(batch_records):,} ({((i + 1)/len(batch_records))*100:.1f}%) | Matches found so far: {total_matches:,}")
                
        print(f"Partition [{country}] completed in {time.time() - c_start:.2f}s.")
        print(f"Candidates generated: {total_cands:,} | Final matches: {total_matches:,}")
        
        # Free memory
        del target_records, blocker, target_dict, s2_chunks, s3_chunks
        
    # 4. Write output files in strict original S1 order
    print("\n" + "=" * 70)
    print("Writing Final Output Files...")
    print("=" * 70)
    
    with open(matching_out_path, "w", encoding="utf-8") as f_match, \
         open(candidate_out_path, "w", encoding="utf-8") as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in df_s1['entity_id']:
            m_list = final_matches_map.get(s1_id, [])
            c_list = final_candidates_map.get(s1_id, [])
            
            f_match.write(f"{s1_id}\t{','.join(m_list)}\n")
            f_cand.write(f"{s1_id}\t{','.join(c_list)}\n")
            
    print(f"Successfully generated:")
    print(f"  1. {matching_out_path} ({os.path.getsize(matching_out_path):,} bytes)")
    print(f"  2. {candidate_out_path} ({os.path.getsize(candidate_out_path):,} bytes)")
    print(f"Total time elapsed: {time.time() - start_total:.2f}s")
    print("Done!")

if __name__ == "__main__":
    # If run directly with an argument, allow testing on small slice or full
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Sample size of S1 to process (None for full)")
    args = parser.parse_args()
    
    run_test_inference(sample_size=args.sample)
