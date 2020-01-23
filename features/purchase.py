import logging

import pandas as pd

from features.utils import drop_column_multi_index_inplace

logger = logging.getLogger(__name__)

ORDER_COLUMNS = [
    'transaction_id',
    'datetime',
    'regular_points_received',
    'express_points_received',
    'regular_points_spent',
    'express_points_spent',
    'purchase_sum',
    'store_id',
]

def make_purchase_features(purchases: pd.DataFrame)-> pd.DataFrame:
    # Purchase is one row in bill. Order is a whole bill.

    logger.info('Creating purchase features...')

    logger.info('Creating really purchase features...')
    purchase_features = make_really_purchase_features(purchases)
    logger.info('Really purchase features are created')


    logger.info('Preparing orders table...')

    orders = purchases.reindex(columns=['client_id'] + ORDER_COLUMNS)
    del purchases
    orders.drop_duplicates(inplace=True)
    logger.info(f'Orders table is ready. Orders: {len(orders)}')

    logger.info('Creating order features...')
    order_features = make_order_features(orders)
    logger.info('Order features are created')

    features = pd.merge(
        purchase_features,
        order_features,
        on='client_id'
    )

    logger.info('Purchase features are created')
    return features


def make_really_purchase_features(purchases: pd.DataFrame) -> pd.DataFrame:
    p_gb = purchases.groupby(['client_id', 'transaction_id'])
    purchase_agg = p_gb.agg(
        {
            'product_id': ['count'],
            'product_quantity': ['max']
        }
    )
    drop_column_multi_index_inplace(purchase_agg)
    purchase_agg.reset_index(inplace=True)
    o_gb = purchase_agg.groupby('client_id')
    features = o_gb.agg(
        {
            'product_id_count': ['mean'],  # mean products in order
            'product_quantity_max': ['mean'],  # mean max number of one product
        }
    )
    drop_column_multi_index_inplace(features)
    features.reset_index(inplace=True)

    return features


def make_order_features(orders: pd.DataFrame) -> pd.DataFrame:
    o_gb = orders.groupby('client_id')
    features = o_gb.agg(
        {
            'transaction_id': ['count'],  # number of orders
            'regular_points_received': ['sum', 'max', 'median'],
            'express_points_received': ['sum', 'max', 'median'],
            'regular_points_spent': ['sum', 'min', 'median'],
            'express_points_spent': ['sum', 'min', 'median'],
            'purchase_sum': ['sum', 'max', 'median'],
            'store_id': ['nunique'],  # number of unique stores
        }
    )
    drop_column_multi_index_inplace(features)
    features.reset_index(inplace=True)
    return features
