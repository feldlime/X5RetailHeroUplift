import os
from os.path import join as pjoin

PROJECT_PATH = os.path.dirname(__file__)
DATA_PATH = pjoin(PROJECT_PATH, 'data')
SUBMISSIONS_PATH = pjoin(PROJECT_PATH, 'submissions')

RANDOM_STATE = 1

# Values for debug
N_PURCHASES_ROWS = 200000
N_ALS_ITERATIONS = 15
