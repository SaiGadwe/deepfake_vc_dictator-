import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes the Equal Error Rate (EER), optimal threshold, and ROC-AUC.
    
    In voice authenticity & ASVspoof benchmarking:
    - y_true: Ground truth binary labels (0 = Human / Bonafide, 1 = Synthetic / Spoof)
    - y_scores: Predicted probabilities P(fake) in [0.0, 1.0]
    
    Returns:
        eer: Equal Error Rate [0.0 - 1.0]
        optimal_threshold: The decision threshold theta where FAR == FRR
        auc: Area Under the ROC Curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Calculate EER where fpr == fnr (i.e. fpr + fnr == 1 -> tpr + fpr == 1)
    try:
        eer = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr, fill_value="extrapolate")(x), 0.0, 1.0))
        optimal_threshold = float(interp1d(fpr, thresholds, fill_value="extrapolate")(eer))
    except Exception:
        # Fallback to minimum absolute difference
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        optimal_threshold = float(thresholds[idx])

    auc = float(roc_auc_score(y_true, y_scores))
    return eer, optimal_threshold, auc

def generate_evaluation_report(y_true: np.ndarray, y_scores: np.ndarray, custom_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Generates a full statistical evaluation summary for ASVspoof 5 offline model benchmarks.
    """
    eer, opt_thresh, auc = compute_eer(y_true, y_scores)
    
    # Evaluate at custom operational threshold (e.g. 0.5 or 0.65)
    y_pred = (y_scores >= custom_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    far = float(np.mean((y_scores >= custom_threshold) & (y_true == 0)))  # False alarms on human
    frr = float(np.mean((y_scores < custom_threshold) & (y_true == 1)))   # Missed AI deepfakes
    
    return {
        "eer": round(eer, 4),
        "eer_percentage": f"{eer * 100:.2f}%",
        "optimal_threshold": round(opt_thresh, 4),
        "roc_auc": round(auc, 4),
        "operational_threshold": custom_threshold,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "false_alarm_rate_far": round(far, 4),
        "miss_rate_frr": round(frr, 4),
        "total_evaluation_samples": int(len(y_true))
    }
