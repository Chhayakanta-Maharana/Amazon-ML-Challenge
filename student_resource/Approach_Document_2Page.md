# Amazon ML Challenge 2026: Business Entity Resolution Technical Report

**Team:** Arcade | **Institution:** Government College of Engineering, Kalahandi  
**Team Members:** Mohit Kabi, Debabrata Pradhan, Hari Pangi, Chhayakanta Maharana  
**Date:** 26 September 2026 | **Validation Macro $F_{0.5}$:** **0.9934** | **Candidates:** **~36.5 / entity**

---

## 1. Executive Summary
We propose a high-throughput, precision-calibrated two-stage pipeline for enterprise-scale multi-source Business Entity Resolution across multilingual records (United States, India, and an unseen country, France). The system employs a 4-channel hybrid inverted-index blocking mechanism, reducing an initial 17.2 trillion pairwise comparison space by >99.9997% to an average of just ~36.5 candidates per entity (strictly $\le 50$). These candidates are evaluated using an 8-dimensional non-linear feature extractor and a HistGradientBoosting classifier, with the decision threshold optimized to 0.85 for Macro $F_{0.5}$ precision-oriented optimization. The pipeline operates 100% offline with zero external APIs/databases. On internal validation, it achieves a **Macro $F_{0.5}$ score of 0.9934** (Precision: 99.58%, Recall: 98.40%), processing all 1.73M test entities at ~470 entities/second on CPU.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Severe Textual & Suffix Noise:** High variance in legal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `Inc`, `SARL`, `SA`), abbreviations, spelling typos, and word-order transpositions (`"Apollo Pharmacy"` vs `"Pharmacy Apollo"`).
- **Address Non-Standardization:** High frequency of missing postal codes, colloquial landmark-based Indian addresses, street abbreviations (`Rd`, `St`, `Blvd`), and unstructured unit/suite numbering.
- **Domain Shift (Unseen France):** Training includes US and India, while test introduces France (`FR`). To avoid out-of-vocabulary degradation, feature extraction relies on language-agnostic mathematical string distances.
- **Metric Asymmetry & Singleton Trap:** Macro $F_{0.5}$ weights precision $2\times$ over recall. With >68% singletons (0 matches in Source 1), predicting even a single false match gives 0.0 for that entity, demanding high-precision thresholding.

### 2.2 Solution Strategy & Pipeline Architecture
`Raw Inputs (S1, S2, S3) -> Country Normalization -> 4-Channel Inverted Blocking -> 8-Dim Feature Vector -> HistGradientBoosting (tau=0.85) -> Validated TSVs`

- **Country-Partitioned Streaming:** Execution runs per country (US, IN, FR). This isolates comparisons, prevents cross-border false matches, and bounds peak RAM <2.5 GB.
- **Vectorized Mini-Batch Scoring:** Evaluates candidates in 2,000-entity chunks (~60,000 pairs), achieving ~470 entities/second (31x speedup over iterative inference).
- **Precision Threshold Tuning ($\tau=0.85$):** Raising $\tau$ from 0.50 to 0.85 suppresses borderline false positives, yields 100% singleton accuracy, and lifts Macro $F_{0.5}$ from 0.9854 to 0.9934.

---

## 3. Candidate Generation (Blocking)
To reduce 1.73M $\times$ 9.97M $\approx$ 17.27 trillion comparisons to $\le 50$ candidates/entity, we deploy 4 complementary inverted indexing channels:
1. **5-Character Alphanumeric Prefix:** Normalized name prefix to capture frontal alignment and resist suffix typos.
2. **Sorted Non-Stopword Tokens:** Lexicographically sorted tokens ensure word-order invariance (`"Cafe Coffee Day"` $\equiv$ `"Coffee Day Cafe"`).
3. **Distinctive Name Token Index:** Inverted index for rare, specific tokens; common tokens appearing in >5,000 records (`services`, `store`, `solutions`) are pruned to prevent fanout explosions.
4. **Compound Address-Numeric Key:** Combines building/postal numbers with street tokens to link distinct trade names at identical physical locations.

> **Blocking Performance:** Generated **63,334,920 candidates** across 1,732,544 test entities (**~36.55 candidates/entity**, strictly $\le 50$ cap). Union of channels yields **>99.4% candidate recall ceiling** while pruning >99.9997% of comparisons.

---

## 4. Matching Model & Feature Engineering

### 4.1 8-Dimensional Dense Feature Vector
- **Name Similarities:** RapidFuzz Token Sort Ratio (word order), Partial Ratio (substring/acronym containment), Character Levenshtein ratio, Token Jaccard overlap, and Name Length Difference ratio.
- **Address Similarities:** Address Token Set Ratio (handles extra landmark/suite words), Address Partial Substring ratio, and Exact Postal/PIN code match indicator.
- **Exact Matching & Signals:** Exact normalized phone number match flag and multi-channel blocking agreement count.

### 4.2 Model Type & Training
- **Classifier:** `HistGradientBoostingClassifier` (scikit-learn, 100 trees, `learning_rate=0.08`, `max_depth=7`, `min_samples_leaf=20`, `l2_reg=0.1`). Parameter count <1M (<1 MB disk size, well within the 8B constraint; BSD/MIT license).
- **Optimal Threshold ($\tau = 0.85$):** Determined via grid search over validation set directly optimizing entity-level Macro $F_{0.5}$ with singleton penalization.

---

## 5. Results & Error Analysis
Evaluated on a 25% held-out stratified validation split (7,500 Source 1 entities with full candidate background pools):

| Threshold ($\tau$) | Macro $F_{0.5}$ | Precision | Recall | Macro $F_1$ | Singleton Acc. |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 0.9854 | 0.9842 | 0.9901 | 0.9871 | 98.4% |
| 0.70 | 0.9912 | 0.9926 | 0.9875 | 0.9900 | 99.2% |
| **0.85 (Optimal)** | **0.9934** | **0.9958** | **0.9840** | **0.9899** | **100.0%** |
| 0.90 | 0.9901 | 0.9968 | 0.9650 | 0.9806 | 100.0% |

- **False Positives (Wrong Merges):** Primarily multi-branch retail chains sharing identical brand names in adjacent postal sectors with missing unit numbers. Resolved via strict 0.85 threshold and postal consistency.
- **False Negatives (Missed Links):** Rare corner cases where an unexpanded acronym is combined with an unpopulated or colloquial landmark address with zero shared tokens.
- **Singleton Preservation:** 1,178,637 test entities (68.0%) are singletons. Threshold 0.85 eliminated false merges, securing full 1.0 credit per singleton.
- **Validator Check:** `utils/validate_submission.py` passed all 1,732,544 rows: **PASS — no blocking issues found. Safe to submit.**

---

## 6. Conclusion, Code Artefacts & Compliance
Our solution pairs multi-key candidate blocking with language-agnostic fuzzy string features and precision-calibrated gradient boosting. It runs 100% offline without external APIs/databases, satisfies the $\le 50$ candidate limit (~36.55 achieved), and delivers a validation Macro $F_{0.5}$ score of **0.9934** across 1.73M entities.

- **Source Code:** `code/business_entity_resolution/src/` (`config`, `preprocessing`, `blocking`, `features`, `model`, `train_pipeline`, `run_inference`, `evaluation`).
- **Reproduction:** `pip install -r requirements.txt` $\to$ `python src/train_pipeline.py` $\to$ `python src/run_inference.py`.
- **Runtime Profile:** Train: ~65s | Inference: ~470 entities/s | Peak RAM: <2.5 GB | Model Size: 570 KB.

---

## References
1. P. Christen. *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection*. Springer, 2012.
2. I. P. Fellegi and A. B. Sunter. "A Theory for Record Linkage." *JASA*, 64(328):1183–1210, 1969.
3. G. Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 30*, 2017.
4. F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *JMLR*, 12:2825–2830, 2011.
5. G. Navarro. "A Guided Tour to Approximate String Matching." *ACM CSUR*, 33(1):31–88, 2001.
6. C. J. van Rijsbergen. *Information Retrieval*. Butterworth-Heinemann, 2nd Edition, 1979.
