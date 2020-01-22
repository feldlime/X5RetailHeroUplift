import logging
from os.path import join as pjoin
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import DATA_PATH

logger = logging.getLogger(__name__)


def load_clients() -> pd.DataFrame:
    return pd.read_csv(
        pjoin(DATA_PATH, 'clients.csv'),
        parse_dates=['first_issue_date', 'first_redeem_date'],
    )


def prepare_clients() -> Tuple[pd.DataFrame, LabelEncoder]:
    logger.info('Preparing clients...')
    clients = load_clients()
    client_encoder = LabelEncoder()
    clients['client_id'] = client_encoder.fit_transform(clients['client_id'])
    logger.info('Clients are ready')
    return clients, client_encoder


def load_products() -> pd.DataFrame:
    return pd.read_csv(pjoin(DATA_PATH, 'products.csv'))


def prepare_products() -> Tuple[pd.DataFrame, LabelEncoder]:
    logger.info('Preparing products')
    products = load_products()
    product_encoder = LabelEncoder()
    products['product_id'] = product_encoder.\
        fit_transform(products['product_id'])

    products = products.fillna(-1)

    for col in [
        'level_1', 'level_2', 'level_3', 'level_4',
        'segment_id', 'brand_id', 'vendor_id',
    ]:
        products[col] = LabelEncoder().fit_transform(products[col].astype(str))
    logger.info('Products are ready')
    return products, product_encoder


def load_purchases() -> pd.DataFrame:
    logger.info('Loading purchases...')
    purchases = pd.read_csv(
        pjoin(DATA_PATH, 'purchases.csv')
    )
    logger.info('Purchases are loaded')
    return purchases


def prepare_purchases(
    client_encoder: LabelEncoder,
    product_encoder: LabelEncoder,
) -> pd.DataFrame:
    logger.info('Preparing purchases...')
    purchases = load_purchases()

    logger.info('Handling n/a values...')
    purchases.dropna(subset=['client_id', 'product_id'], how='any')
    purchases.fillna(-1)

    logger.info('Label encoding...')
    purchases['client_id'] = client_encoder.transform(purchases['client_id'])
    purchases['product_id'] = product_encoder.transform(purchases['product_id'])
    for col in ['transaction_id', 'store_id']:
        purchases[col] = LabelEncoder().\
            fit_transform(purchases[col].fillna(-1).astype(str))

    logger.info('Date and time conversion...')
    purchases['datetime'] = pd.to_datetime(
        purchases['transaction_datetime'],
        format='%Y-%m-%d %H:%M:%S',
    )
    purchases.drop(columns=['transaction_datetime'], inplace=True)

    logger.info('Purchases are ready')
    return purchases


def load_train() -> pd.DataFrame:
    return pd.read_csv(
        pjoin(DATA_PATH, 'uplift_train.csv'),
        index_col='client_id',
    )


def load_test() -> pd.DataFrame:
    return pd.read_csv(
        pjoin(DATA_PATH, 'uplift_test.csv'),
        index_col='client_id',
    )
