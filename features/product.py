import os
import logging

import numpy as np
import pandas as pd
import time

from config import N_ALS_ITERATIONS
from .utils import drop_column_multi_index_inplace, make_latent_feature

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

logger = logging.getLogger(__name__)

N_FACTORS = {
    'product_id': 32,
    'level_1': 2,
    'level_2': 3,
    'level_3': 4,
    'level_4': 5,
    'segment_id': 4,
    'brand_id': 10,
    'vendor_id': 10,
}
#
# N_FACTORS = {
#     'product_id': 64,
#     'level_1': 2,
#     'level_2': 4,
#     'level_3': 8,
#     'level_4': 8,
#     'segment_id': 8,
#     'brand_id': 8,
#     'vendor_id': 8,
# }

N_ITERATIONS = N_ALS_ITERATIONS


def make_product_features(
    products: pd.DataFrame,
    purchases: pd.DataFrame,
) -> pd.DataFrame:
    """
        From purchases need only columns:
        - client_id
        - product_id

        Columns client_id and product_id must be label encoded!
    """

    logger.info('Creating purchases-products matrix')
    purchases_products = pd.merge(
        purchases,
        products,
        on='product_id',
    )
    logger.info('Purchases-products matrix is ready')

    # Aliases only
    del purchases
    del products

    logger.info('Creating latent features')
    latent_features = make_latent_features(purchases_products)

    logger.info('Creating usual features')
    usual_features = make_usual_features(purchases_products)

    logger.info('Combining features')
    features = pd.merge(
        latent_features,
        usual_features,
        on='client_id'
    )

    logger.info(f'Product features are created. Shape = {features.shape}')
    return features


def make_usual_features(
    purchases_products: pd.DataFrame,
) -> pd.DataFrame:
    pp_gb = purchases_products.groupby('client_id')
    usual_features = pp_gb.agg(
        {
            'netto': 'median',
            'is_own_trademark': ['sum', 'mean'],
            'is_alcohol': ['sum', 'mean'],
        }
    )
    drop_column_multi_index_inplace(usual_features)
    usual_features.reset_index(inplace=True)

    return usual_features


def make_latent_features(
    purchases_products: pd.DataFrame,
) -> pd.DataFrame:
    latent_feature_matrices = []
    latent_feature_names = []
    for col, n_factors in N_FACTORS.items():
        logger.info(f'Creating latent features for {col}')

        counts_subject_by_client = (
            purchases_products
            .groupby('client_id')[col]
            .transform('count')
        )
        share_col = f'{col}_share'
        purchases_products[share_col] = 1 / counts_subject_by_client

        latent_feature_matrices.append(
            make_latent_feature(
                purchases_products,
                index_col='client_id',
                value_col=col,
                n_factors=n_factors,
                n_iterations=N_ITERATIONS,
                sum_col=share_col,
            )
        )

        purchases_products.drop(columns=share_col, inplace=True)

        latent_feature_names.extend(
            [f'{col}_f{i+1}' for i in range(n_factors)]
        )

        logger.info(f'Features for {col} were created')

    # Add features that show how much client likes product in category
    logger.info(f'Creating latent features for product in 4th category')
    col = 'product_id'
    counts_products_by_client_and_category = (
        purchases_products
            .groupby(['client_id', 'level_4'])[col]
            .transform('count')
    )
    share_col = f'{col}_share_by_client_and_cat'
    purchases_products[share_col] = 1 / counts_products_by_client_and_category

    n_factors = N_FACTORS[col]
    latent_feature_matrices.append(
        make_latent_feature(
            purchases_products,
            index_col='client_id',
            value_col=col,
            n_factors=n_factors,
            n_iterations=N_ITERATIONS,
            sum_col=share_col,
        )
    )

    purchases_products.drop(columns=share_col, inplace=True)

    latent_feature_names.extend(
        [f'product_by_cat_f{i+1}' for i in range(n_factors)]
    )

    logger.info(f'Features for {col} were created in sec')

    latent_features = pd.DataFrame(
        np.hstack(latent_feature_matrices),
        columns=latent_feature_names
    )
    latent_features.insert(0, 'client_id', np.arange(latent_features.shape[0]))

    return latent_features
