from os.path import join as pjoin

import pandas as pd

from config import SUBMISSIONS_PATH


def save_submission(indices_test, test_pred, filename):
    df_submission = pd.DataFrame({'uplift': test_pred}, index=indices_test)
    df_submission.to_csv(pjoin(SUBMISSIONS_PATH, filename))
