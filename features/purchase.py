import logging
from datetime import timedelta

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import N_ALS_ITERATIONS, MAILING_DATETIME
from features.utils import (
    drop_column_multi_index_inplace,
    make_count_csr,
    SECONDS_IN_DAY,
    make_latent_feature,
)

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

FLOAT32_MAX = np.finfo(np.float32).max
POINT_TYPES = ('regular', 'express')
POINT_EVENT_TYPES = ('spent', 'received')
WEEK_DAYS = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]
TIME_LABELS = ['Night', 'Morning', 'Afternoon', 'Evening']


def make_purchase_features_for_last_days(
    purchases: pd.DataFrame,
    n_days: int
) -> pd.DataFrame:
    logger.info(f'Creating purchase features for last {n_days} days...')
    cutoff = MAILING_DATETIME - timedelta(days=n_days)
    purchases_last = purchases[purchases['datetime'] >= cutoff]
    purchase_last_features = make_purchase_features(purchases_last)
    logger.info(f'Purchase features for last {n_days} days are created')
    return purchase_last_features


def make_purchase_features(purchases: pd.DataFrame) -> pd.DataFrame:
    # Purchase is one row in bill. Order is a whole bill.

    logger.info('Creating purchase features...')

    n_clients = purchases['client_id'].nunique()

    logger.info('Creating really purchase features...')
    purchase_features = make_really_purchase_features(purchases)
    logger.info('Really purchase features are created')

    logger.info('Creating small product features...')
    product_features = make_small_product_features(purchases)
    logger.info('Small product features are created')

    logger.info('Preparing orders table...')

    orders = purchases.reindex(columns=['client_id'] + ORDER_COLUMNS)
    del purchases
    orders.drop_duplicates(inplace=True)
    logger.info(f'Orders table is ready. Orders: {len(orders)}')

    logger.info('Creating order features...')
    order_features = make_order_features(orders)
    logger.info('Order features are created')

    logger.info('Creating time features...')
    time_features = make_time_features(orders)
    logger.info('Time features are created')

    logger.info('Creating store features...')
    store_features = make_store_features(orders)
    logger.info('Store features are created')

    logger.info('Creating order interval features...')
    order_interval_features = make_order_interval_features(orders)
    logger.info('Order interval features are created')

    features = (
        purchase_features
        .merge(order_features, on='client_id')
        .merge(time_features, on='client_id')
        .merge(product_features, on='client_id')
        .merge(store_features, on='client_id')
        .merge(order_interval_features, on='client_id')
    )

    logger.info('Creating ratio time features...')
    ratio_time_features = make_ratio_time_features(features)
    logger.info('Ratio time features are created')

    features = features.merge(ratio_time_features, on='client_id')

    assert len(features) == n_clients, \
        f'n_clients = {n_clients} but len(features) = {len(features)}'

    features['days_from_last_order_share'] = \
        features['days_from_last_order'] / features['orders_interval_median']

    features['most_popular_store_share'] = (
        features['store_transaction_id_count_max'] /
        features['transaction_id_count']
    )

    logger.info(f'Purchase features are created. Shape = {features.shape}')
    return features


def make_really_purchase_features(purchases: pd.DataFrame) -> pd.DataFrame:
    simple_purchases = purchases.reindex(
        columns=['client_id', 'product_id', 'trn_sum_from_iss']
    )
    prices_bounds = [0, 98, 195, 490, 950, 1900, 4400, FLOAT32_MAX]
    agg_dict = {}
    for i, lower_bound in enumerate(prices_bounds[:-1]):
        upper_bound = prices_bounds[i + 1]
        name = f'price_from_{lower_bound}'
        simple_purchases[name] = (
            (simple_purchases['trn_sum_from_iss'] >= lower_bound) &
            (simple_purchases['trn_sum_from_iss'] < upper_bound)
        ).astype(int)
        agg_dict[name] = ['sum', 'mean']

    agg_dict.update(
        {
            'trn_sum_from_iss': ['median'],  # median product price
            'product_id': ['count', 'nunique'],
        }
    )
    simple_features = simple_purchases.groupby('client_id').agg(agg_dict)
    drop_column_multi_index_inplace(simple_features)
    simple_features.reset_index(inplace=True)

    p_gb = purchases.groupby(['client_id', 'transaction_id'])
    purchase_agg = p_gb.agg(
        {
            'product_id': ['count'],
            'product_quantity': ['max'],
        }
    )
    drop_column_multi_index_inplace(purchase_agg)
    purchase_agg.reset_index(inplace=True)
    o_gb = purchase_agg.groupby('client_id')
    complex_features = o_gb.agg(
        {
            # mean products in order
            'product_id_count': ['mean', 'median'],
            # mean max number of one product
            'product_quantity_max': ['mean', 'median'],
        }
    )
    drop_column_multi_index_inplace(complex_features)
    complex_features.reset_index(inplace=True)

    features = pd.merge(
        simple_features,
        complex_features,
        on='client_id'
    )
    return features


def make_order_features(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()

    o_gb = orders.groupby('client_id')

    agg_dict = {
            'transaction_id': ['count'],  # number of orders
            'regular_points_received': ['sum', 'max', 'median'],
            'express_points_received': ['sum', 'max', 'median'],
            'regular_points_spent': ['sum', 'min', 'median'],
            'express_points_spent': ['sum', 'min', 'median'],
            'purchase_sum': ['sum', 'max', 'median'],
            'store_id': ['nunique'],  # number of unique stores
            'datetime': ['max'],  # datetime of last order
        }

    # is regular/express points spent/received
    for points_type in POINT_TYPES:
        for event_type in POINT_EVENT_TYPES:
            col_name = f'{points_type}_points_{event_type}'
            new_col_name = f'is_{points_type}_points_{event_type}'
            orders[new_col_name] = (orders[col_name] != 0).astype(int)
            agg_dict[new_col_name] = ['sum']

    features = o_gb.agg(agg_dict)
    drop_column_multi_index_inplace(features)
    features.reset_index(inplace=True)

    features['days_from_last_order'] = (
        MAILING_DATETIME - features['datetime_max']
    ).dt.total_seconds() // SECONDS_IN_DAY
    features.drop(columns=['datetime_max'], inplace=True)

    # proportion of regular/express points spent to all transactions
    for points_type in POINT_TYPES:
        for event_type in POINT_EVENT_TYPES:
            col_name = f'is_{points_type}_points_{event_type}_sum'
            new_col_name = f'proportion_count_{points_type}_points_{event_type}'
            features[new_col_name] = (
                    features[col_name] / features['transaction_id_count']
            )

    express_col = f'is_express_points_spent_sum'
    regular_col = f'is_regular_points_spent_sum'
    new_col_name = f'ratio_count_express_to_regular_points_spent'
    features[new_col_name] = (
            features[express_col] / features[regular_col]
    ).replace(np.inf, FLOAT32_MAX)

    for points_type in POINT_TYPES:
        spent_col = f'is_{points_type}_points_spent_sum'
        received_col = f'is_{points_type}_points_received_sum'
        new_col_name = f'ratio_count_{points_type}_points_spent_to_received'
        features[new_col_name] = (
                features[spent_col] / features[received_col]
        ).replace(np.inf, 1000)

    return features


def make_time_features(orders: pd.DataFrame) -> pd.DataFrame:
    # np.unique returns sorted array
    client_ids = np.unique(orders['client_id'].values)

    orders['weekday'] = np.array(WEEK_DAYS)[
        orders['datetime'].dt.dayofweek.values
    ]

    time_bins = [-1, 6, 11, 18, 24]

    orders['part_of_day'] = pd.cut(
        orders['datetime'].dt.hour,
        bins=time_bins,
        labels=TIME_LABELS,
    ).astype(str)

    time_part_encoder = LabelEncoder()
    orders['part_of_day'] = time_part_encoder.fit_transform(orders['part_of_day'])

    time_part_columns_name = time_part_encoder.inverse_transform(
        np.arange(len(time_part_encoder.classes_))
    )

    time_part_cols = make_count_csr(
        orders,
        index_col='client_id',
        value_col='part_of_day',
    )[client_ids, :]  # drop empty rows

    time_part_cols = pd.DataFrame(
        time_part_cols.toarray(),
        columns=time_part_columns_name,
    )
    time_part_cols['client_id'] = client_ids

    weekday_encoder = LabelEncoder()
    orders['weekday'] = weekday_encoder.fit_transform(orders['weekday'])

    weekday_column_names = weekday_encoder.inverse_transform(
        np.arange(len(weekday_encoder.classes_))
    )
    weekday_cols = make_count_csr(
        orders,
        index_col='client_id',
        value_col='weekday',  # weekday time part
    )[client_ids, :]  # drop empty rows
    weekday_cols = pd.DataFrame(
        weekday_cols.toarray(),
        columns=weekday_column_names,
    )
    weekday_cols['client_id'] = client_ids

    time_part_features = pd.merge(
        left=time_part_cols,
        right=weekday_cols,
        on='client_id',
    )
    time_part_features.columns = [
        f'{col}_orders_count' if col != 'client_id' else col
        for col in time_part_features.columns
    ]

    return time_part_features


def make_small_product_features(purchases: pd.DataFrame) -> pd.DataFrame:
    cl_pr_gb = purchases.groupby(['client_id', 'product_id'])
    product_agg = cl_pr_gb.agg({
        'product_quantity': ['sum'],
    })

    drop_column_multi_index_inplace(product_agg)
    product_agg.reset_index(inplace=True)

    cl_gb = product_agg.groupby(['client_id'])
    features = cl_gb.agg({'product_quantity_sum': ['max']})

    drop_column_multi_index_inplace(features)
    features.reset_index(inplace=True)

    return features


def make_store_features(orders: pd.DataFrame) -> pd.DataFrame:
    cl_st_gb = orders.groupby(['client_id', 'store_id'])
    store_agg = cl_st_gb.agg({
        'transaction_id': ['count'],
    })

    drop_column_multi_index_inplace(store_agg)
    store_agg.reset_index(inplace=True)

    cl_gb = store_agg.groupby(['client_id'])
    simple_features = cl_gb.agg(
        {
            'transaction_id_count': ['max', 'mean', 'median']
        }
    )

    drop_column_multi_index_inplace(simple_features)
    simple_features.reset_index(inplace=True)
    simple_features.columns = (
        ['client_id'] +
        [
            f'store_{col}'
            for col in simple_features.columns[1:]
        ]
    )

    latent_features = make_latent_store_features(orders)

    features = pd.merge(
        simple_features,
        latent_features,
        on='client_id'
    )

    return features


def make_latent_store_features(orders: pd.DataFrame) -> pd.DataFrame:
    n_factors = 8
    latent_feature_names = [f'store_id_f{i + 1}' for i in range(n_factors)]

    latent_feature_matrix = make_latent_feature(
        orders,
        index_col='client_id',
        value_col='store_id',
        n_factors=n_factors,
        n_iterations=N_ALS_ITERATIONS,
    )

    latent_features = pd.DataFrame(
        latent_feature_matrix,
        columns=latent_feature_names
    )
    latent_features.insert(0, 'client_id', np.arange(latent_features.shape[0]))

    return latent_features


def make_order_interval_features(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.sort_values(['client_id', 'datetime'])

    last_order_client = orders['client_id'].shift(1)
    is_same_client = last_order_client == orders['client_id']
    orders['last_order_datetime'] = orders['datetime'].shift(1)

    orders['orders_interval'] = np.nan
    orders.loc[is_same_client, 'orders_interval'] = (
        orders.loc[is_same_client, 'datetime'] -
        orders.loc[is_same_client, 'last_order_datetime']
    ).dt.total_seconds() / SECONDS_IN_DAY

    cl_gb = orders.groupby('client_id', sort=False)
    features = cl_gb.agg(
        {
            'orders_interval': [
                'mean',  # mean interval between orders
                'median',
                'std',  # constancy of orders
                'min',
                'max',
                'last',  # interval between last 2 orders
            ]
        }
    )
    drop_column_multi_index_inplace(features)
    features.reset_index(inplace=True)
    features.fillna(-3, inplace=True)

    return features


def make_ratio_time_features(features: pd.DataFrame) -> pd.DataFrame:
    time_labels = TIME_LABELS + WEEK_DAYS
    columns = [f'{col}_orders_count' for col in time_labels]
    share_columns = [f'{col}_share' for col in columns]

    time_features = features.reindex(columns=columns).values
    orders_count = features['transaction_id_count'].values

    share_time_features = time_features / orders_count.reshape(-1, 1)

    share_time_features = pd.DataFrame(
        share_time_features,
        columns=share_columns,
    )
    share_time_features['client_id'] = features['client_id']

    return share_time_features
