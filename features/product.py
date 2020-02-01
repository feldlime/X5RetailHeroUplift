import os
import logging

import numpy as np
import pandas as pd
import time
from scipy import sparse

from implicit.als import AlternatingLeastSquares

from config import RANDOM_STATE, N_ALS_ITERATIONS
from features.utils import drop_column_multi_index_inplace

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

    # Aliases only
    del purchases
    del products

    logger.info('Creating latent features')
    latent_features = make_latent_features(purchases_products)

    logger.info('Creating usual features')
    usual_features = make_usual_features(purchases_products)

    logger.info('Combining features')
    product_features = pd.merge(
        latent_features,
        usual_features,
        on='client_id'
    )

    return product_features


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
        start_time = time.time()

        latent_feature_matrices.append(
            make_latent_feature(
                purchases_products,
                col,
                n_factors,
                N_ITERATIONS,
            )
        )

        latent_feature_names.extend(
            [f'{col}_f{i+1}' for i in range(n_factors)])

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f'Features for {col} were created in {elapsed:.1f} sec')

    latent_features = pd.DataFrame(
        np.hstack(latent_feature_matrices),
        columns=latent_feature_names
    )
    latent_features.insert(0, 'client_id', np.arange(latent_features.shape[0]))

    return latent_features


def make_latent_feature(
    df: pd.DataFrame,
    col: str,
    n_factors: int,
    iterations: int,
):
    csr = make_count_csr(df, col)

    model = AlternatingLeastSquares(
        factors=n_factors,
        dtype=np.float32,
        iterations=iterations,
        regularization=0.1,
        use_gpu=True if n_factors >= 32 else False,

    )
    np.random.seed(RANDOM_STATE)
    model.fit(csr)

    return model.user_factors


def make_count_csr(
        df: pd.DataFrame,
        value_col: str,
        col_index_col: str = 'client_id',
) -> sparse.csr_matrix:
    coo = sparse.coo_matrix(
        (
            np.ones(len(df)),
            (
                df[value_col].values,
                df[col_index_col].values
            )
        )
    )
    csr = coo.tocsr(copy=False)
    return csr


