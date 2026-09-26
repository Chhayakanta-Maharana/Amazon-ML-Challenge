# Amazon ML Challenge 2026: Business Entity Resolution Technical Report

**Team:** Arcade  
**Institution:** Government College of Engineering, Kalahandi  
**Technical Report:** Business Entity Resolution Solution Template  
**Team Members:** Mohit Kabi, Debabrata Pradhan, Hari Pangi, Chhayakanta Maharana  
**Date:** 26 September 2026  
**GitHub Repository:** [https://github.com/Chhayakanta-Maharana/Amazon-ML-Challenge](https://github.com/Chhayakanta-Maharana/Amazon-ML-Challenge)

---

## 1. Executive Summary

We present a high-throughput, precision-optimized Machine Learning pipeline for large-scale multi-source Business Entity Resolution across multilingual and multi-national enterprise data (US, India, and France). Our solution integrates a **4-channel hybrid blocking architecture** (name prefix, sorted n-gram tokens, distinctive vocabulary, and compound address-numeric keys) with a **25-dimensional non-linear feature extractor** and a **HistGradientBoosting Classifier** fine-tuned on entity-level **Macro $F_{0.5}$** optimization ($Threshold = 0.85$). The pipeline operates in streaming, country-partitioned batches to process ~10 million records with high memory efficiency, achieving an internal validation Macro $F_{0.5}$ score of **0.9934** (Precision: 0.9958, Recall: 0.9840).

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis (EDA) across training and test splits, we identified key structural challenges:
1. **Name Noise & Permutations:** Extreme frequency of company suffix variations (`Pvt Ltd`, `Private Limited`, `Corp`, `Inc`, `LLC`, `SARL`, `SA`), phonetic transliteration noise, punctuation disparities (`&` vs `and`), and token order permutations.
2. **Address Inconsistencies:** Landmark-oriented Indian addresses, missing PIN/postal codes, street abbreviations (`Rd`, `Blvd`, `St`, `Ave`), and unstructured apartment/suite numbering.
3. **Domain Shift & Unseen Countries:** The training set consists of `US` and `India`, whereas the test set introduces `France`. To ensure zero out-of-vocabulary penalty, our text normalization and feature engineering strictly avoid hardcoded country-specific vocabularies or one-hot encodings.
4. **Extreme Asymmetry of Evaluation Metric ($F_{0.5}$):** With $\beta = 0.5$, precision is penalized twice as heavily as recall. False positive merges severely degrade macro scores; singletons (entities with 0 matches) must be preserved cleanly.

### 2.2 Solution Strategy
We developed a two-stage **Blocking + Gradient Boosting Classifier** pipeline:
- **Approach Type:** Hybrid Multi-Channel Inverted Index Blocking + Pairwise Gradient Boosted Tree Classifier with Entity-Level Decision Thresholding.
- **Core Innovation:** 
  1. *Country-Partitioned Streaming*: Partitioning candidate generation and inference per country isolate comparison spaces naturally, eliminating $O(N \times M)$ memory footprint and cross-border false positives.
  2. *Channel-Frequency Awareness*: Candidate generation records how many distinct channels triggered a pair, feeding this directly as an engineered feature into the classifier.
  3. *Exact Macro $F_{0.5}$ Optimization*: Custom threshold sweep optimizing the exact metric including singleton penalization.

```
+-----------------------------------------------------------------------------------+
|                            Raw Input Records (S1, S2, S3)                         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               Country-Aware Unicode & Entity-Type Normalization                   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         Stage 1: Multi-Channel Blocking                           |
|  * 5-char Alphanumeric Prefix Key                                                 |
|  * Sorted Token-Pair Key (Order-Invariant)                                        |
|  * Low-Frequency Distinctive Name Token Index                                     |
|  * Numeric Address + Street Keyword Compound Key                                 |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|             Stage 2: 25-Dimensional Pairwise Feature Extraction                   |
|  * String & Token Similarity (Jaro-Winkler, Levenshtein, Token Sort/Set Ratios)   |
|  * Token Jaccard Overlaps (Name & Address)                                        |
|  * Address Numeric / Postal Code Set Intersection                                 |
|  * Multi-Channel Candidate Confidence Score                                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               Stage 3: HistGradientBoosting Inference (Thresh = 0.85)             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|       Validated Submission (matching_results.tsv & candidate_pairs.tsv)           |
+-----------------------------------------------------------------------------------+
```

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.97\text{M} \approx 1.72 \times 10^{13}$ all-pairs search space to a strictly manageable candidate pool without dropping true matches, we utilize 4 complementary inverted indexing channels:

1. **Alphanumeric Name Prefix Key:** First 5 alphanumeric characters after legal suffix removal (captures typos at name endings and prefix alignment).
2. **Sorted First 2 Non-Stopword Tokens:** Lexicographically sorted token pairs (ensures invariance against word-order transpositions like *"Apollo Pharmacy"* vs *"Pharmacy Apollo"*).
3. **Distinctive Name Token Inverted Index:** Filters out top generic stop-words (e.g., `services`, `enterprises`, `international`, `group`) and indexes high-specificity lexical tokens with fanout ceiling $\le 150$.
4. **Compound Address-Numeric Key:** Combines numerical identifiers (building numbers, postal codes) with primary street tokens to capture entities with variant trade names at identical physical addresses.

- **Candidate pairs generated:** 63,334,920 candidate pairs across 1,732,544 test entities (Average ~36.55 candidates per Source 1 entity, strictly complying with the $\le 50$ competition cap).
- **Ensuring Zero Lost Matches (Recall Ceiling):** True match recall ceiling achieved across validation sets exceeds **99.4%** while reducing the candidate search space by $>99.9997\%$.

---

## 4. Matching Model

### 4.1 Features Used (25 Total Dimensions)
- **Name Similarity Features:**
  - Character Levenshtein ratio (`name_ratio`)
  - Token Sort Ratio & Token Set Ratio (`name_sort_ratio`, `name_set_ratio`)
  - Jaro-Winkler Similarity (`name_jw`)
  - Token Jaccard Overlap (`name_jaccard`)
  - Normalized Length Disparity (`name_len_diff`)
  - Exact Binary Match (`name_exact`)
- **Address Similarity Features:**
  - Full address edit distance & token set similarity (`addr_ratio`, `addr_set_ratio`)
  - Address Jaro-Winkler & Token Jaccard (`addr_jw`, `addr_jaccard`)
  - Length disparity and exact match flags (`addr_len_diff`, `addr_exact`)
  - Numeric & PIN code intersection ratio (`num_overlap`, `num_exact`)
- **Structural & Interaction Features:**
  - Country match verification (`same_country`)
  - Field presence flags (`has_name_both`, `has_addr_both`)
  - Multi-channel blocking agreement count (`channel_count`)
  - Non-linear name $\times$ address interaction (`name_x_addr`, `weighted_sim`, `both_high`)

### 4.2 Model Type & Optimization
- **Model Type:** `HistGradientBoostingClassifier` (scikit-learn, MIT License, $<1\text{M}$ params, zero external GPU dependency).
  - Hyperparameters: `max_iter=160`, `learning_rate=0.07`, `max_depth=7`, `min_samples_leaf=20`, `l2_regularization=0.1`.
- **Threshold Selection:** Exhaustive grid search over validation set directly optimizing the macro-averaged $F_{0.5}$ metric across all entities including singletons. The optimal threshold was determined at **$\tau = 0.85$**, which strictly suppresses false positive merges in accordance with the precision-weighted metric.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
Evaluated on a stratified 25% held-out validation split (7,500 Source 1 entities with full background candidate pools):

| Threshold | Macro $F_{0.5}$ | Precision | Recall | Macro $F_1$ |
| :--- | :---: | :---: | :---: | :---: |
| 0.50 | 0.9854 | 0.9842 | 0.9901 | 0.9871 |
| 0.70 | 0.9912 | 0.9926 | 0.9875 | 0.9900 |
| **0.85 (Optimal)** | **0.9934** | **0.9958** | **0.9840** | **0.9899** |
| 0.90 | 0.9901 | 0.9968 | 0.9650 | 0.9806 |

### 5.2 Error Analysis
- **False Positives (Wrong Merges):** Primarily occur when two distinct retail branches of large chains (e.g., franchises or supermarket chains) share identical brand names in adjacent street addresses with missing door numbers. Mitigated by strict numeric token matching and a high threshold ($0.85$).
- **False Negatives (Missed Matches):** Occur in rare edge cases where both business name and address suffer simultaneous extreme corruption (e.g. acronym-only name combined with a purely landmark-based address lacking any shared street tokens).

---

## 6. Conclusion
Our solution demonstrates that combining multi-channel domain-agnostic inverted index blocking with rapid non-linear gradient boosted feature matching delivers superior accuracy and scalability. The country-partitioned architecture natively scales to millions of records while maintaining strict precision, achieving a validation Macro $F_{0.5}$ of **0.9934** with full compliance to competition constraints and zero external data lookup.

---

## Appendix

### A. Code Artefacts & Reproduction Guide

The complete source code is organized as follows:
```
code/business_entity_resolution/
├── src/
│   ├── config.py              # Configuration & file paths
│   ├── preprocessing.py       # Unicode normalization & regex cleaning
│   ├── blocking.py           # Multi-channel candidate generation index
│   ├── features.py           # 25-dimensional pairwise similarity extraction
│   ├── model.py              # HistGradientBoosting classifier wrapper
│   ├── train_pipeline.py     # End-to-end training & threshold optimization
│   ├── run_inference.py      # High-throughput test set inference
│   └── evaluation.py         # Exact Macro F0.5 metric implementation
├── requirements.txt          # Pinned environment dependencies
└── README.md                 # Step-by-step reproduction instructions
```

**Reproduction Steps:**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train model & optimize threshold:
   ```bash
   python src/train_pipeline.py
   ```
3. Generate test candidate pairs and final matches:
   ```bash
   python src/run_inference.py
   ```
4. Validate generated submission files:
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```

### B. Hardware & Runtime Profile
- **Training Time:** ~65 seconds on standard multi-core CPU.
- **Inference Throughput:** ~470 entities / second (vectorized mini-batch inference).
- **Peak Memory Footprint:** $< 2.5\text{ GB}$ RAM via compact string tuples and streaming country-partitioned batching.
- **Model Size:** 570 KB on disk ($< 1\text{M}$ parameters, far below the 8B constraint).

---

## References

1. **Peter Christen.** *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection.* Data-Centric Systems and Applications, Springer Science & Business Media, 2012.
2. **Ivan P. Fellegi and Alan B. Sunter.** "A Theory for Record Linkage." *Journal of the American Statistical Association*, 64(328):1183–1210, 1969.
3. **Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu.** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems (NeurIPS 30)*, 2017.
4. **Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, et al.** "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research (JMLR)*, 12:2825–2830, 2011.
5. **Gonzalo Navarro.** "A Guided Tour to Approximate String Matching." *ACM Computing Surveys (CSUR)*, 33(1):31–88, 2001.
6. **C. J. van Rijsbergen.** *Information Retrieval.* Butterworth-Heinemann, 2nd Edition, 1979. (Formulation of the parameterized $F_\beta$ and $F_{0.5}$ metric).
