import os

class Config:
    # Random Seed for Reproducibility
    SEED = 42
    
    # Base paths relative to student_resource/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
    
    # Train files
    TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
    TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
    TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
    TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    
    # Test files
    TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
    TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
    TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")
    
    # Output files
    MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    
    # Model & Feature Hyperparameters
    MAX_CANDIDATES_PER_S1 = 50
    TFIDF_MAX_FEATURES = 10000
    
    # Threshold search space
    DEFAULT_THRESHOLD = 0.50
