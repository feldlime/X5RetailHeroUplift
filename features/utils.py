import pandas as pd

def drop_column_multi_index_inplace(df: pd.DataFrame) -> None:
    df.columns = ['_'.join(t) for t in df.columns]
