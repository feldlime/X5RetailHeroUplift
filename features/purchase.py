import logging

import pandas as pd

logger = logging.getLogger(__name__)


def make_purchase_features(purchases: pd.DataFrame)-> pd.DataFrame:

    p_gb = purchases.groupby('client_id')

    features = p_gb['transaction_id'].nunique()
    features.reset_index(inplace=True)
    features.rename(columns={'transaction_id': 'n_purchases'})

    return features
