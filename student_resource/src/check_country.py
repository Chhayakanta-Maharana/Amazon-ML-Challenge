import os
import pandas as pd

def check_country_consistency():
    base_dir = "student_resource/dataset"
    print("Checking country consistency across ground truth...")
    
    s1 = pd.read_csv(os.path.join(base_dir, "train/train_source1.tsv"), sep="\t", usecols=['entity_id', 'country'])
    s2 = pd.read_csv(os.path.join(base_dir, "train/train_source2.tsv"), sep="\t", usecols=['entity_id', 'country'])
    s3 = pd.read_csv(os.path.join(base_dir, "train/train_source3.tsv"), sep="\t", usecols=['entity_id', 'country'])
    gt = pd.read_csv(os.path.join(base_dir, "train/train_ground_truth.tsv"), sep="\t").dropna(subset=['matched_entity_ids'])
    
    s1_country = dict(zip(s1['entity_id'], s1['country']))
    s23_country = dict(zip(s2['entity_id'], s2['country']))
    s23_country.update(dict(zip(s3['entity_id'], s3['country'])))
    
    cross_country_matches = 0
    total_pairs = 0
    
    # Sample 100,000 GT rows to check speed and consistency
    sample_gt = gt.sample(n=min(100000, len(gt)), random_state=42)
    for _, row in sample_gt.iterrows():
        s1_id = row['source1_entity_id']
        c1 = s1_country.get(s1_id)
        for m_id in str(row['matched_entity_ids']).split(','):
            m_id = m_id.strip()
            if m_id:
                total_pairs += 1
                c2 = s23_country.get(m_id)
                if c1 != c2:
                    cross_country_matches += 1
                    
    print(f"Total sampled true pairs: {total_pairs}")
    print(f"Cross-country matches: {cross_country_matches} ({cross_country_matches/max(1, total_pairs)*100:.4f}%)")

if __name__ == "__main__":
    check_country_consistency()
