from os.path import join as pjoin
import os

import pandas as pd


PROJECT_PATH = os.path.dirname(__file__)
DATA_PATH = pjoin(PROJECT_PATH, 'data')
SUBMISSIONS_PATH = pjoin(PROJECT_PATH, 'submissions')

RANDOM_STATE = 1


# Values for debug
N_PURCHASES_ROWS = 1_000  # None
N_ALS_ITERATIONS = 2  # 15
N_ESTIMATORS = 10


def save_submission(indices_test, test_pred, filename):
    df_submission = pd.DataFrame({'uplift': test_pred}, index=indices_test)
    df_submission.to_csv(pjoin(SUBMISSIONS_PATH, filename))
