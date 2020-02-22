import logging

import pandas as pd

from .utils import SECONDS_IN_DAY

logger = logging.getLogger(__name__)

INTERVALS = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'hour']

def make_client_features(clients: pd.DataFrame) -> pd.DataFrame:
    """No id in index"""

    logger.info('Preparing features')
    min_datetime = clients['first_issue_date'].min()

    days_from_min_to_issue = (
            (clients['first_issue_date'] - min_datetime)
            .dt.total_seconds() /
            SECONDS_IN_DAY
    ).values
    days_from_min_to_redeem = (
            (clients['first_redeem_date'] - min_datetime)
            .dt.total_seconds() /
            SECONDS_IN_DAY
    ).values

    age = clients['age'].values
    age[age < 0] = -2
    age[age > 100] = -3

    gender = clients['gender'].values

    logger.info('Combining features')
    features = pd.DataFrame({
        'client_id': clients['client_id'].values,
        'gender_M': (gender == 'M').astype(int),
        'gender_F': (gender == 'F').astype(int),
        'gender_U': (gender == 'U').astype(int),
        'age': age,
        'days_from_min_to_issue': days_from_min_to_issue,
        'days_from_min_to_redeem': days_from_min_to_redeem,
        'issue_redeem_delay': days_from_min_to_redeem - days_from_min_to_issue,
    })
    # for event in ['issue', 'redeem']:
    #     for interval in INTERVALS:
    #         values = getattr(clients[f'first_{event}_date'].dt, interval)
    #         features[f'{event}_{interval}'] = values

    features = features.fillna(-1)

    logger.info(f'Client features are created. Shape = {features.shape}')
    return features
