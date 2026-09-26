import os
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from preprocessing import clean_name, clean_address, clean_country
from blocking import Blocker
from features import extract_pairwise_features, FEATURE_NAMES
from model import EntityMatcherModel
from evaluation import evaluate_macro_f05, compute_entity_f05

def run_training_and_validation():
    print("="*70)
    print("PHASES 8 - 26: TRAINING, VALIDATION & THRESHOLD OPTIMIZATION")
    print("="*70)
    
    start_time = time.time()
    
    # 1. Load Training Data Samples
    print("\n[Step 1] Loading representative training records...")
    df_s1 = pd.read_csv(Config.TRAIN_S1, sep="\t", nrows=30000)
    s1_ids_set = set(df_s1['entity_id'])
    
    print("Loading ground truth mapping...")
    # Load ground truth and filter to our S1 slice
    gt_map = {}
    all_needed_target_ids = set()
    for chunk in pd.read_csv(Config.TRAIN_GT, sep="\t", chunksize=200000):
        sub_chunk = chunk[chunk['source1_entity_id'].isin(s1_ids_set)]
        for row in sub_chunk.itertuples(index=False):
            s1_id = row.source1_entity_id
            matches_str = str(row.matched_entity_ids)
            if pd.notna(row.matched_entity_ids) and matches_str.strip():
                m_list = [m.strip() for m in matches_str.split(',') if m.strip()]
                gt_map[s1_id] = set(m_list)
                all_needed_target_ids.update(m_list)
            else:
                gt_map[s1_id] = set()
        if len(gt_map) == len(s1_ids_set):
            break
            
    # For any s1_id not found in gt, default to empty set
    for eid in s1_ids_set:
        if eid not in gt_map:
            gt_map[eid] = set()
            
    print(f"Loaded {len(df_s1)} Source 1 entities with ground truth.")
    print(f"Identified {len(all_needed_target_ids)} distinct target S2/S3 match records needed for GT.")
    
    # Load S2 and S3 target pools efficiently
    print("\n[Step 2] Loading target S2/S3 pools (with GT targets + background pool)...")
    # Read chunked S2 and S3
    s2_chunks = []
    s3_chunks = []
    
    for chunk in pd.read_csv(Config.TRAIN_S2, sep="\t", chunksize=100000):
        s2_chunks.append(chunk[chunk['entity_id'].isin(all_needed_target_ids)])
        if sum(len(c) for c in s2_chunks) > 0.9 * len([x for x in all_needed_target_ids if x.startswith('S2-')]) and len(s2_chunks) > 5:
            break
            
    for chunk in pd.read_csv(Config.TRAIN_S3, sep="\t", chunksize=100000):
        s3_chunks.append(chunk[chunk['entity_id'].isin(all_needed_target_ids)])
        if sum(len(c) for c in s3_chunks) > 0.9 * len([x for x in all_needed_target_ids if x.startswith('S3-')]) and len(s3_chunks) > 5:
            break
            
    # Also grab general background records from S2 and S3 head
    bg_s2 = pd.read_csv(Config.TRAIN_S2, sep="\t", nrows=100000)
    bg_s3 = pd.read_csv(Config.TRAIN_S3, sep="\t", nrows=100000)
    
    target_pool = pd.concat(s2_chunks + s3_chunks + [bg_s2, bg_s3], ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"Combined candidate target pool size: {len(target_pool)} records.")
    
    # Preprocess all records
    print("\n[Step 3] Preprocessing S1 and Target records...")
    s1_dict = {}
    for row in df_s1.itertuples(index=False):
        s1_dict[row.entity_id] = {
            'country': clean_country(row.country),
            'name': clean_name(row.business_name),
            'addr': clean_address(row.business_address)
        }
        
    target_dict = {}
    for row in target_pool.itertuples(index=False):
        target_dict[row.entity_id] = {
            'country': clean_country(row.country),
            'name': clean_name(row.business_name),
            'addr': clean_address(row.business_address)
        }
        
    # Build Blocker index
    print("\n[Step 4] Building Blocker Index...")
    blocker = Blocker(max_candidates_per_entity=50)
    blocker.index_target_records(target_pool)
    
    # Entity-Level Train / Validation Split (80% train / 20% val)
    print("\n[Step 5] Creating Leakage-Safe Entity-Level Train/Validation Split...")
    s1_ids = list(df_s1['entity_id'])
    # Stratify by whether entity is a singleton or not
    singleton_labels = [1 if len(gt_map[eid]) == 0 else 0 for eid in s1_ids]
    train_eids, val_eids = train_test_split(s1_ids, test_size=0.25, random_state=Config.SEED, stratify=singleton_labels)
    
    print(f"Train Source 1 Entities: {len(train_eids)}")
    print(f"Validation Source 1 Entities: {len(val_eids)}")
    
    # Construct Training Pairs (Positives + Hard Negatives from Blocking)
    print("\n[Step 6] Constructing Training Pairs & Hard Negatives...")
    X_train = []
    y_train = []
    
    pos_count = 0
    neg_count = 0
    
    for eid in train_eids:
        s1_rec = s1_dict[eid]
        true_matches = gt_map.get(eid, set())
        
        # 1. Add all true positive pairs present in pool
        for tm in true_matches:
            if tm in target_dict:
                t_rec = target_dict[tm]
                feat = extract_pairwise_features(
                    s1_rec['name'], s1_rec['addr'], s1_rec['country'],
                    t_rec['name'], t_rec['addr'], t_rec['country'],
                    channel_count=3
                )
                X_train.append(feat)
                y_train.append(1)
                pos_count += 1
                
        # 2. Add hard negatives from Blocker
        cands, ch_hits = blocker.generate_candidates_for_entity(
            eid, s1_rec['name']['raw'], s1_rec['addr']['raw'], s1_rec['country']
        )
        
        # Mine negatives: candidates that are NOT in true_matches
        for cand in cands:
            if cand not in true_matches and cand in target_dict:
                t_rec = target_dict[cand]
                feat = extract_pairwise_features(
                    s1_rec['name'], s1_rec['addr'], s1_rec['country'],
                    t_rec['name'], t_rec['addr'], t_rec['country'],
                    channel_count=len(ch_hits[cand])
                )
                X_train.append(feat)
                y_train.append(0)
                neg_count += 1
                
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Training dataset: {len(X_train)} pairs (Positives: {pos_count}, Negatives: {neg_count})")
    
    # Train Model
    print("\n[Step 7] Training HistGradientBoosting Matching Model...")
    matcher = EntityMatcherModel(random_state=Config.SEED)
    matcher.fit(X_train, y_train)
    
    # Feature Importances / Separability Analysis
    print(f"Model successfully trained. Feature count: {len(FEATURE_NAMES)}")
    
    # Validation & Exact Macro F0.5 Threshold Optimization
    print("\n[Step 8] Running Full Validation Pipeline across Validation Set...")
    val_gt = {eid: gt_map[eid] for eid in val_eids}
    
    val_candidate_preds = {} # {s1_id: [(cand_id, prob)]}
    
    for eid in val_eids:
        s1_rec = s1_dict[eid]
        cands, ch_hits = blocker.generate_candidates_for_entity(
            eid, s1_rec['name']['raw'], s1_rec['addr']['raw'], s1_rec['country']
        )
        
        cand_list = []
        if cands:
            cand_features = []
            valid_cands = []
            for cand in cands:
                if cand in target_dict:
                    t_rec = target_dict[cand]
                    f = extract_pairwise_features(
                        s1_rec['name'], s1_rec['addr'], s1_rec['country'],
                        t_rec['name'], t_rec['addr'], t_rec['country'],
                        channel_count=len(ch_hits[cand])
                    )
                    cand_features.append(f)
                    valid_cands.append(cand)
                    
            if cand_features:
                probs = matcher.predict_proba(np.array(cand_features))
                for cand_id, prob in zip(valid_cands, probs):
                    cand_list.append((cand_id, float(prob)))
                    
        val_candidate_preds[eid] = cand_list
        
    print(f"Validation inference complete for {len(val_eids)} entities.")
    
    # Threshold Grid Search
    print("\n[Step 9] Optimizing Decision Threshold for Entity-Level Macro F0.5...")
    thresholds = np.arange(0.15, 0.95, 0.05)
    best_thresh = 0.50
    best_f05 = -1.0
    best_metrics = {}
    
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 62)
    
    for th in thresholds:
        preds = {}
        for eid, c_list in val_candidate_preds.items():
            matches = [c_id for c_id, p in c_list if p >= th]
            preds[eid] = matches
            
        metrics = evaluate_macro_f05(preds, val_gt)
        print(f"{th:<10.2f} | {metrics['macro_f05']:<12.4f} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1_score']:<10.4f}")
        
        if metrics['macro_f05'] > best_f05:
            best_f05 = metrics['macro_f05']
            best_thresh = th
            best_metrics = metrics
            
    print("-" * 62)
    print(f"\n>>> Optimal Threshold: {best_thresh:.2f}")
    print(f">>> Best Validation Macro F0.5: {best_metrics['macro_f05']:.4f}")
    print(f">>> Validation Precision:       {best_metrics['precision']:.4f}")
    print(f">>> Validation Recall:          {best_metrics['recall']:.4f}")
    print(f">>> Validation F1 Score:        {best_metrics['f1_score']:.4f}")
    
    matcher.best_threshold = best_thresh
    model_save_path = os.path.join(Config.ARTIFACTS_DIR, "matching_model.joblib")
    matcher.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    print(f"Pipeline executed in {time.time() - start_time:.2f}s")
    
    return best_metrics

if __name__ == "__main__":
    run_training_and_validation()
