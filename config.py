import os
from datetime import datetime
from os.path import join as pjoin

PROJECT_PATH = os.path.dirname(__file__)
DATA_PATH = pjoin(PROJECT_PATH, 'data')
SUBMISSIONS_PATH = pjoin(PROJECT_PATH, 'submissions')

MAILING_DATETIME = datetime(2019, 3, 19)

RANDOM_STATE = 1

# Values for debug
N_PURCHASES_ROWS = None
N_ALS_ITERATIONS = 15
