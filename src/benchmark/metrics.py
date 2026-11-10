import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Optional, Tuple, Union

def compute_apcer(y_true: np.ndarray, y_pred: np.ndarray, attack_types: Optional[np.ndarray] = None) -> float:
    """
    Compute Attack Presentation Classification Error Rate (APCER).
    Convention: Bona Fide = 0, Attack = 1.
    APCER is the proportion of attack presentations classified as bona fide.
    If attack_types is provided, computes APCER per attack type and returns the maximum.
    """
    if len(y_true) == 0:
        return 0.0

    if attack_types is not None:
        apcer_list = []
        unique_attacks = np.unique(attack_types[y_true == 1])
        for at in unique_attacks:
            mask = (attack_types == at) & (y_true == 1)
            if np.sum(mask) == 0:
                continue
            errors = np.sum((y_pred[mask] == 0))
            apcer_list.append(errors / np.sum(mask))
        return max(apcer_list) if apcer_list else 0.0
    else:
        attack_mask = (y_true == 1)
        if np.sum(attack_mask) == 0:
            return 0.0
        return float(np.sum(y_pred[attack_mask] == 0) / np.sum(attack_mask))

def compute_bpcer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Bona Fide Presentation Classification Error Rate (BPCER).
    Convention: Bona Fide = 0, Attack = 1.
    BPCER is the proportion of bona fide presentations classified as attacks.
    """
    bona_fide_mask = (y_true == 0)
    if np.sum(bona_fide_mask) == 0:
        return 0.0
    return float(np.sum(y_pred[bona_fide_mask] == 1) / np.sum(bona_fide_mask))

def compute_acer(apcer: float, bpcer: float) -> float:
    """
    Compute Average Classification Error Rate (ACER).
    """
    return (apcer + bpcer) / 2.0

def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (AUC).
    """
    if len(np.unique(y_true)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return float(auc(fpr, tpr))

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    """
    if len(np.unique(y_true)) < 2:
        return float('nan')
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)

def compute_det_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Detection Error Tradeoff (DET) curve points.
    Returns: (FPR, FNR, thresholds)
    """
    if len(np.unique(y_true)) < 2:
        return np.array([]), np.array([]), np.array([])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    return fpr, fnr, thresholds

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, attack_types: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all PAD metrics and return them in a dictionary.
    """
    apcer = compute_apcer(y_true, y_pred, attack_types)
    bpcer = compute_bpcer(y_true, y_pred)
    acer = compute_acer(apcer, bpcer)
    auc_val = compute_auc(y_true, y_scores)
    eer = compute_eer(y_true, y_scores)

    return {
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "AUC": auc_val,
        "EER": eer
    }
