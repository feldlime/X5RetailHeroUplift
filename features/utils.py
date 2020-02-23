from typing import Optional

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse

from config import RANDOM_STATE

SECONDS_IN_DAY = 60 * 60 * 24


def drop_column_multi_index_inplace(df: pd.DataFrame) -> None:
    df.columns = ['_'.join(t) for t in df.columns]


def make_count_csr(
    df: pd.DataFrame,
    index_col: str,
    value_col: str,
) -> sparse.csr_matrix:
    col_to_sum_name = '__col_to_sum__'
    df['__col_to_sum__'] = 1
    csr = make_sum_csr(
        df,
        index_col=index_col,
        value_col=value_col,
        col_to_sum=col_to_sum_name,
    )
    df.drop(columns=col_to_sum_name, inplace=True)
    return csr


def make_sum_csr(
    df: pd.DataFrame,
    index_col: str,
    value_col: str,
    col_to_sum: str,
) -> sparse.csr_matrix:
    coo = sparse.coo_matrix(
        (
            df[col_to_sum].values,
            (
                df[index_col].values,
                df[value_col].values,
            )
        )
    )
    csr = coo.tocsr(copy=False)
    return csr


def make_latent_feature(
    df: pd.DataFrame,
    index_col: str,
    value_col: str,
    n_factors: int,
    n_iterations: int,
    sum_col: Optional[str] = None
):
    if sum_col is None:
        csr = make_count_csr(df, index_col=index_col, value_col=value_col)
    else:
        csr = make_sum_csr(
            df,
            index_col=index_col,
            value_col=value_col,
            col_to_sum=sum_col,
        )

    model = AlternatingLeastSquares(
        factors=n_factors,
        dtype=np.float32,
        iterations=n_iterations,
        regularization=0.1,
        use_gpu=False,  # True if n_factors >= 32 else False,

    )
    np.random.seed(RANDOM_STATE)
    model.fit(csr.T)

    return model.user_factors
