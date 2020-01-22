from os.path import join as pjoin
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import DATA_PATH


def load_clients() -> pd.DataFrame:
    return pd.read_csv(
        pjoin(DATA_PATH, 'clients.csv'),
        parse_dates=['first_issue_date', 'first_redeem_date'],
    )


def prepare_clients() -> Tuple[pd.DataFrame, LabelEncoder]:
    clients = load_clients()
    client_encoder = LabelEncoder()
    clients['client_id'] = client_encoder.fit_transform(clients['client_id'])
    return clients, client_encoder


def load_products() -> pd.DataFrame:
    return pd.read_csv(pjoin(DATA_PATH, 'products.csv'))


def prepare_products() -> Tuple[pd.DataFrame, LabelEncoder]:
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
    return products, product_encoder


def load_purchases() -> pd.DataFrame:
    return pd.read_csv(
        pjoin(DATA_PATH, 'purchases.csv')
    )


def prepare_purchases(
    client_encoder: LabelEncoder,
    product_encoder: LabelEncoder,
) -> pd.DataFrame:
    purchases = load_purchases()

    purchases['client_id'] = client_encoder.transform(purchases['client_id'])
    purchases['product_id'] = product_encoder.transform(purchases['product_id'])

    purchases.dropna(subset=['client_id', 'product_id'], how='any')
    purchases.fillna(-1)

    for col in ['transaction_id', 'store_id']:
        purchases[col] = LabelEncoder().\
            fit_transform(purchases[col].fillna(-1).astype(str))

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
