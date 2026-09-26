import numpy as np

def compute_entity_f05(pred_matches, true_matches):
    """
    Computes macro F_0.5 score for a single Source 1 entity according to official specification.
    pred_matches: set or list of predicted matched_entity_ids
    true_matches: set or list of ground truth matched_entity_ids
    """
    pred_set = set(pred_matches) if pred_matches else set()
    true_set = set(true_matches) if true_matches else set()
    
    # Singleton case: No true matches
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
        
    # Non-singleton case with no predictions
    if len(pred_set) == 0:
        return 0.0
        
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision == 0.0 and recall == 0.0:
        return 0.0
        
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

def evaluate_macro_f05(predictions_dict, ground_truth_dict):
    """
    Evaluates macro F_0.5 across all Source 1 entities in ground truth.
    predictions_dict: {s1_id: set/list of predicted IDs}
    ground_truth_dict: {s1_id: set/list of true IDs}
    """
    f05_scores = []
    precisions = []
    recalls = []
    
    for s1_id, true_set in ground_truth_dict.items():
        pred_set = set(predictions_dict.get(s1_id, []))
        
        # Calculate individual entity F0.5
        score = compute_entity_f05(pred_set, true_set)
        f05_scores.append(score)
        
        # Track individual precision & recall for diagnostic metrics
        if len(true_set) > 0 and len(pred_set) > 0:
            tp = len(pred_set & true_set)
            precisions.append(tp / len(pred_set))
            recalls.append(tp / len(true_set))
        elif len(true_set) == 0 and len(pred_set) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
        elif len(true_set) == 0 and len(pred_set) > 0:
            precisions.append(0.0)
            recalls.append(1.0)
        else: # len(true_set) > 0 and len(pred_set) == 0
            precisions.append(0.0)
            recalls.append(0.0)
            
    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    
    # Global F1 calculation
    if (avg_precision + avg_recall) > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0.0
        
    return {
        "macro_f05": macro_f05,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": f1_score,
        "num_entities": len(ground_truth_dict)
    }
