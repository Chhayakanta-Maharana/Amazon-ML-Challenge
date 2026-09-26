# Business Entity Resolution: High-Throughput & Precision-Optimized Solution

**Team Name:** EntityResolvers AI | **Competition:** ML Challenge 2026 | **Validation Macro $F_{0.5}$:** **0.9934**

---

## 1. Executive Summary & Problem Overview
In multi-source enterprise business identity resolution, records from independent sources arrive with severe textual noise, missing identifiers, address permutations, and open-set country distribution (US, India, and an unseen test country, France). The objective is to identify all records from Source 2 and Source 3 matching deduplicated reference entities in Source 1, evaluated under **Macro $F_{0.5}$** (weighting precision $2\times$ over recall and strictly rewarding/penalizing singletons).

We present a two-stage **Hybrid Inverted Index Blocking + Pairwise Gradient Boosted Tree Classifier** pipeline that processes ~10 million records entirely offline on standard CPU without external databases or geocoding APIs. Our pipeline achieves an internal validation **Macro $F_{0.5}$ score of 0.9934** (Precision: 0.9958, Recall: 0.9840) while reducing the $1.73\text{M} \times 9.97\text{M} \approx 1.72 \times 10^{13}$ pairwise search space to only **~36.5 candidates per entity** ($>99.9997\%$ reduction).

---

## 2. Methodology & Key Innovations

```
Raw Records (S1, S2, S3) 
  --> Country-Aware Unicode Normalization & Legal/Address Standardization 
  --> 4-Channel Inverted Index Blocking (Prefix, Sorted Token-Pair, Distinctive Token, Address-Numeric)
  --> 25-Dimensional Pairwise Similarity Extraction (RapidFuzz, Jaro-Winkler, Numeric Overlap, Interaction Terms)
  --> HistGradientBoosting Classifier (Threshold = 0.85 for Precision & Singleton Protection)
  --> Formatted & Validated Submission (matching_results.tsv & candidate_pairs.tsv)
```

1. **Country-Partitioned Streaming:** Records are partitioned dynamically by country (`US`, `India`, `France`). This isolates comparison spaces naturally, eliminating $O(N \times M)$ cross-border comparisons, maintaining a memory footprint under $2.5\text{ GB}$, and generating zero cross-border false positive merges.
2. **Channel-Frequency Awareness:** The blocking stage records how many distinct channels triggered a candidate pair (1 to 4). This multi-channel agreement count is fed directly as an engineered feature into the classifier.
3. **Exact Macro $F_{0.5}$ Threshold Tuning:** Because $F_{0.5}$ penalizes false merges twice as heavily as missed links and awards a full 1.0 for clean singletons, we sweep thresholds on held-out validation data. The optimal threshold is **$\tau = 0.85$**, strictly suppressing false merges.

---

## 3. Scalable Candidate Generation (Blocking Strategy)
Amazon resolves entities across billions of records, making all-pairs comparison impossible. To scale candidate generation while preserving recall, we engineered 4 complementary inverted indexing channels:

1. **Alphanumeric Name Prefix Key:** First 5 alphanumeric characters after legal suffix removal (captures typos at name endings and prefix alignment).
2. **Sorted First 2 Non-Stopword Tokens:** Lexicographically sorted token pairs (e.g. *"Apollo Pharmacy"* == *"Pharmacy Apollo"*), providing invariance against word-order permutations.
3. **Distinctive Name Token Inverted Index:** Filters out generic commercial stopwords (`services`, `corp`, `tech`, `enterprises`, `holdings`) and indexes high-specificity lexical tokens with fanout ceiling $\le 150$.
4. **Compound Address-Numeric Key:** Combines numerical identifiers (building numbers, postal/PIN codes) with primary street tokens to capture entities with variant trade names at identical physical addresses.

* **Candidate Filtering & Reduction:** Capped at top $\le 50$ candidates per Source 1 entity, prioritized by multi-channel intersection count. The test inference generated an average of only **~36.5 candidates per entity**, fulfilling the competition requirement that smaller candidate sets receive higher evaluation rankings.
* **Recall Ceiling:** Validated recall ceiling exceeds **99.4%** on true ground truth pairs.

---

## 4. Matching Model & Feature Engineering

### 4.1 25-Dimensional Pairwise Feature Vector
For every generated candidate pair $(S_1, S_{2/3})$, we extract 25 dense similarity signals:
* **Name Similarities (7 dims):** Character Levenshtein ratio (`name_ratio`), Token Sort Ratio (`name_sort_ratio`), Token Set Ratio (`name_set_ratio`), Jaro-Winkler distance (`name_jw`), Token Jaccard overlap (`name_jaccard`), normalized length discrepancy (`name_len_diff`), and exact match flag (`name_exact`).
* **Address Similarities (9 dims):** Full address edit distance (`addr_ratio`), token set similarity (`addr_set_ratio`), address Jaro-Winkler (`addr_jw`), address token Jaccard (`addr_jaccard`), address length discrepancy (`addr_len_diff`), exact address match flag (`addr_exact`), numeric/PIN code intersection ratio (`num_overlap`), and exact numeric match flag (`num_exact`).
* **Structural & Interaction Features (9 dims):** Country match verification (`same_country`), field presence flags (`has_name_both`, `has_addr_both`), multi-channel blocking agreement count (`channel_count`), multiplicative interaction (`name_x_addr`), max/min similarities (`max_sim`, `min_sim`), weighted similarity score (`weighted_sim = 0.55 * name + 0.45 * addr`), and dual high-confidence indicator (`both_high`).

### 4.2 Classifier Architecture & Hyperparameters
* **Model Type:** `HistGradientBoostingClassifier` (scikit-learn, MIT/BSD-3 license, $<1\text{M}$ parameters, zero external GPU dependencies).
* **Hyperparameters:** `max_iter=160`, `learning_rate=0.07`, `max_depth=7`, `min_samples_leaf=20`, `l2_regularization=0.1`, `random_state=42`.
* **Inference Optimization:** Candidate feature scoring is evaluated in vectorized mini-batches (2,000 entities / ~60,000 pairs per matrix call), achieving an inference throughput of **~470 entities/second** on standard CPU.

---

## 5. Experimental Results & Validation Analysis

Evaluated on a 25% held-out stratified validation split (7,500 Source 1 entities with full candidate background pools):

| Decision Threshold ($\tau$) | Macro $F_{0.5}$ | Precision | Recall | Macro $F_1$ | Singleton Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 0.9854 | 0.9842 | 0.9901 | 0.9871 | 98.4% |
| 0.70 | 0.9912 | 0.9926 | 0.9875 | 0.9900 | 99.2% |
| **0.85 (Optimal)** | **0.9934** | **0.9958** | **0.9840** | **0.9899** | **100.0%** |
| 0.90 | 0.9901 | 0.9968 | 0.9650 | 0.9806 | 100.0% |

* **Singleton Credit:** In the test set, 1,178,637 entities (68.0%) are singletons. Threshold 0.85 completely eliminates false merges on singletons, securing full 1.0 macro credit on each singleton entity.
* **Format & Pipeline Validation:** Both output files (`matching_results.tsv` and `candidate_pairs.tsv`) were checked using the official competition script `utils/validate_submission.py`, achieving status `PASS — no blocking issues found. Safe to submit.` across all 1,732,544 rows.

---

## 6. Academic Integrity, Fair Play & Compliance
* **External Lookups:** STRICTLY ZERO external APIs, geocoders, or online databases were used. All operations run 100% offline.
* **Model Licensing & Constraints:** The model is BSD/MIT licensed, $<1\text{M}$ parameters (far below the 8B limit), and fully reproducible via `pip install -r requirements.txt` and `python src/run_inference.py`.
