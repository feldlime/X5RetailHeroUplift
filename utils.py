from os.path import join as pjoin
import os

import numpy as np


PROJECT_PATH = os.path.dirname(__file__)
DATA_PATH = pjoin(PROJECT_PATH, 'data')
SUBMISSIONS_PATH = pjoin(PROJECT_PATH, 'submissions')




def uplift_score(prediction, treatment, target, rate=0.3):
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
