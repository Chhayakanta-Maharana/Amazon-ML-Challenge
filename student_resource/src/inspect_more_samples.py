import os
import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def inspect_more_samples():
    base_dir = "student_resource/dataset/train"
    print("Loading slices for deep inspection...")
    
    s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep="\t", nrows=100000)
    s2 = pd.read_csv(os.path.join(base_dir, "train_source2.tsv"), sep="\t", nrows=200000)
    s3 = pd.read_csv(os.path.join(base_dir, "train_source3.tsv"), sep="\t", nrows=200000)
    gt = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep="\t", nrows=100000)
    
    s2_dict = s2.set_index('entity_id').to_dict('index')
    s3_dict = s3.set_index('entity_id').to_dict('index')
    s1_dict = s1.set_index('entity_id').to_dict('index')
    
    us_samples = 0
    in_samples = 0
    
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = str(row['matched_entity_ids'])
        if pd.isna(row['matched_entity_ids']) or not matches_str.strip():
            continue
        if s1_id not in s1_dict:
            continue
        s1_rec = s1_dict[s1_id]
        country = s1_rec['country']
        
        matches = [m.strip() for m in matches_str.split(',') if m.strip()]
        matched_records = []
        for m in matches:
            if m in s2_dict:
                matched_records.append((m, s2_dict[m]))
            elif m in s3_dict:
                matched_records.append((m, s3_dict[m]))
                
        if len(matched_records) >= 1:
            if country == 'US' and us_samples < 5:
                us_samples += 1
                print(f"\n[US MATCH {us_samples}] S1: {s1_id}")
                print(f"  S1 Name:    {s1_rec['business_name']}")
                print(f"  S1 Address: {s1_rec['business_address']}")
                for m_id, m_rec in matched_records:
                    print(f"  --> {m_id} Name:    {m_rec['business_name']}")
                    print(f"             Address: {m_rec['business_address']}")
            elif country == 'India' and in_samples < 5:
                in_samples += 1
                print(f"\n[INDIA MATCH {in_samples}] S1: {s1_id}")
                print(f"  S1 Name:    {s1_rec['business_name']}")
                print(f"  S1 Address: {s1_rec['business_address']}")
                for m_id, m_rec in matched_records:
                    print(f"  --> {m_id} Name:    {m_rec['business_name']}")
                    print(f"             Address: {m_rec['business_address']}")
            if us_samples >= 5 and in_samples >= 5:
                break

if __name__ == "__main__":
    inspect_more_samples()
