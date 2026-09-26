import os
import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def inspect_match_patterns():
    base_dir = "student_resource/dataset/train"
    print("Inspecting sample true match patterns...")
    
    s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep="\t", nrows=50000)
    s2 = pd.read_csv(os.path.join(base_dir, "train_source2.tsv"), sep="\t", nrows=100000)
    s3 = pd.read_csv(os.path.join(base_dir, "train_source3.tsv"), sep="\t", nrows=100000)
    gt = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep="\t", nrows=50000)
    
    s2_dict = s2.set_index('entity_id').to_dict('index')
    s3_dict = s3.set_index('entity_id').to_dict('index')
    
    sample_count = 0
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = str(row['matched_entity_ids'])
        if pd.isna(row['matched_entity_ids']) or not matches_str.strip():
            continue
        
        s1_row = s1[s1['entity_id'] == s1_id]
        if s1_row.empty:
            continue
        s1_rec = s1_row.iloc[0]
        
        matches = [m.strip() for m in matches_str.split(',') if m.strip()]
        matched_records = []
        for m in matches:
            if m in s2_dict:
                matched_records.append((m, s2_dict[m]))
            elif m in s3_dict:
                matched_records.append((m, s3_dict[m]))
                
        if len(matched_records) >= 2:
            sample_count += 1
            print(f"\n=================== Sample Match {sample_count} ===================")
            print(f"[S1 Reference] ID: {s1_id} | Country: {s1_rec['country']}")
            print(f"   Name:    {s1_rec['business_name']}")
            print(f"   Address: {s1_rec['business_address']}")
            for m_id, m_rec in matched_records:
                print(f"   --> [{m_id}] Name:    {m_rec['business_name']}")
                print(f"                 Address: {m_rec['business_address']}")
            if sample_count >= 8:
                break

if __name__ == "__main__":
    inspect_match_patterns()
