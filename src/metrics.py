import numpy as np
from typing import Tuple


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and optimal decision threshold.
    """
    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 0.5

    thresholds = np.linspace(min(scores), max(scores), num=1000)
    frr_list = []
    far_list = []

    for t in thresholds:
        frr = np.sum(bonafide_scores >= t) / len(bonafide_scores)
        far = np.sum(spoof_scores < t) / len(spoof_scores)
        frr_list.append(frr)
        far_list.append(far)

    frr_arr = np.array(frr_list)
    far_arr = np.array(far_list)

    diff = np.abs(frr_arr - far_arr)
    min_idx = np.argmin(diff)

    eer = (frr_arr[min_idx] + far_arr[min_idx]) / 2.0 * 100.0
    threshold = thresholds[min_idx]

    return float(eer), float(threshold)


def compute_tdcf(scores: np.ndarray, labels: np.ndarray, p_target: float = 0.05) -> float:
    """
    Computes simplified tandem Detection Cost Function (t-DCF).
    """
    eer, _ = compute_eer(scores, labels)
    tdcf = (eer / 100.0) * (1.0 - p_target)
    return float(tdcf)
