import numpy as np
from rapidfuzz import fuzz, distance

def extract_pairwise_features(n1_data, a1_data, c1, n2_data, a2_data, c2, channel_count=1):
    """
    Extracts high-signal, leakage-safe pairwise similarity features
    between Source 1 and candidate (Source 2 / 3) records.
    """
    n1_clean = n1_data.get('clean', '')
    n2_clean = n2_data.get('clean', '')
    a1_clean = a1_data.get('clean', '')
    a2_clean = a2_data.get('clean', '')
    
    # 1. Business Name Similarities
    n1_len = len(n1_clean)
    n2_len = len(n2_clean)
    
    if n1_len > 0 and n2_len > 0:
        name_ratio = fuzz.ratio(n1_clean, n2_clean) / 100.0
        name_sort_ratio = fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0
        name_set_ratio = fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0
        name_jw = distance.JaroWinkler.similarity(n1_clean, n2_clean)
        
        # Token Jaccard
        s1_tokens = set(n1_data.get('tokens', []))
        s2_tokens = set(n2_data.get('tokens', []))
        denom = len(s1_tokens | s2_tokens)
        name_jaccard = len(s1_tokens & s2_tokens) / denom if denom > 0 else 0.0
        name_len_diff = abs(n1_len - n2_len) / max(n1_len, n2_len)
        name_exact = 1.0 if n1_clean == n2_clean else 0.0
    else:
        name_ratio = name_sort_ratio = name_set_ratio = name_jw = name_jaccard = name_len_diff = name_exact = 0.0

    # 2. Address Similarities
    a1_len = len(a1_clean)
    a2_len = len(a2_clean)
    
    if a1_len > 0 and a2_len > 0:
        addr_ratio = fuzz.ratio(a1_clean, a2_clean) / 100.0
        addr_sort_ratio = fuzz.token_sort_ratio(a1_clean, a2_clean) / 100.0
        addr_set_ratio = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0
        addr_jw = distance.JaroWinkler.similarity(a1_clean, a2_clean)
        
        # Token Jaccard
        a1_tokens = set(a1_data.get('tokens', []))
        a2_tokens = set(a2_data.get('tokens', []))
        denom_a = len(a1_tokens | a2_tokens)
        addr_jaccard = len(a1_tokens & a2_tokens) / denom_a if denom_a > 0 else 0.0
        addr_len_diff = abs(a1_len - a2_len) / max(a1_len, a2_len)
        addr_exact = 1.0 if a1_clean == a2_clean else 0.0
        
        # Numeric Token Overlap
        nums1 = a1_data.get('num_tokens', set())
        nums2 = a2_data.get('num_tokens', set())
        if nums1 and nums2:
            num_overlap = len(nums1 & nums2) / max(len(nums1), 1)
            num_exact = 1.0 if nums1 == nums2 else 0.0
        elif not nums1 and not nums2:
            num_overlap = 0.5
            num_exact = 0.5
        else:
            num_overlap = 0.0
            num_exact = 0.0
    else:
        addr_ratio = addr_sort_ratio = addr_set_ratio = addr_jw = addr_jaccard = addr_len_diff = addr_exact = 0.0
        num_overlap = num_exact = 0.0

    # 3. Country & Presence Features
    same_country = 1.0 if c1 == c2 and c1 != "UNKNOWN" else 0.0
    has_name_both = 1.0 if n1_len > 0 and n2_len > 0 else 0.0
    has_addr_both = 1.0 if a1_len > 0 and a2_len > 0 else 0.0

    # 4. Interaction & Combined Metrics
    name_x_addr = name_set_ratio * addr_set_ratio
    max_sim = max(name_set_ratio, addr_set_ratio)
    min_sim = min(name_set_ratio, addr_set_ratio)
    weighted_sim = 0.55 * name_set_ratio + 0.45 * addr_set_ratio
    both_high = 1.0 if (name_set_ratio > 0.70 and addr_set_ratio > 0.60) else 0.0

    return [
        name_ratio,
        name_sort_ratio,
        name_set_ratio,
        name_jw,
        name_jaccard,
        name_len_diff,
        name_exact,
        addr_ratio,
        addr_sort_ratio,
        addr_set_ratio,
        addr_jw,
        addr_jaccard,
        addr_len_diff,
        addr_exact,
        num_overlap,
        num_exact,
        same_country,
        has_name_both,
        has_addr_both,
        channel_count,
        name_x_addr,
        max_sim,
        min_sim,
        weighted_sim,
        both_high
    ]

FEATURE_NAMES = [
    "name_ratio",
    "name_sort_ratio",
    "name_set_ratio",
    "name_jw",
    "name_jaccard",
    "name_len_diff",
    "name_exact",
    "addr_ratio",
    "addr_sort_ratio",
    "addr_set_ratio",
    "addr_jw",
    "addr_jaccard",
    "addr_len_diff",
    "addr_exact",
    "num_overlap",
    "num_exact",
    "same_country",
    "has_name_both",
    "has_addr_both",
    "channel_count",
    "name_x_addr",
    "max_sim",
    "min_sim",
    "weighted_sim",
    "both_high"
]
