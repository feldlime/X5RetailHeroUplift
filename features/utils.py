import numpy as np
import pandas as pd
from scipy import sparse


def drop_column_multi_index_inplace(df: pd.DataFrame) -> None:
    df.columns = ['_'.join(t) for t in df.columns]


def make_count_csr(
        df: pd.DataFrame,
        value_col: str,
        col_index_col: str = 'client_id',
) -> sparse.csr_matrix:
    coo = sparse.coo_matrix(
        (
            np.ones(len(df)),
            (
                df[col_index_col].values,
                df[value_col].values,
            )
        )
    )
    csr = coo.tocsr(copy=False)
    return csr


def make_sum_csr(
        df: pd.DataFrame,
        value_col: str,
        col_to_sum: str,
        col_index_col: str = 'client_id',
) -> sparse.csr_matrix:
    coo = sparse.coo_matrix(
        (
            df[col_to_sum].values,
            (
                df[col_index_col].values,
                df[value_col].values,
            )
        )
    )
    csr = coo.tocsr(copy=False)
    return csr


