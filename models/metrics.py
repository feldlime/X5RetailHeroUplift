from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from models.utils import make_z


def score_uplift(
    prediction: np.ndarray,
    treatment: np.ndarray,
    target: np.ndarray,
    rate: float = 0.3,
) -> float:
    """
    Подсчет Uplift Score
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score


def score_roc_auc(
    prediction: np.ndarray,
    treatment: np.ndarray,
    target: np.ndarray,
) -> float:
    y_true = make_z(treatment, target)
    score = roc_auc_score(y_true, prediction)
    return score


def uplift_metrics(
    prediction: np.ndarray,
    treatment: np.ndarray,
    target: np.ndarray,
    rate_for_uplift: float = 0.3,
) -> Dict[str, float]:
    scores = {
        'roc_auc': score_roc_auc(prediction, treatment, target),
        'uplift': score_uplift(prediction, treatment, target, rate_for_uplift),
    }
    return scores
