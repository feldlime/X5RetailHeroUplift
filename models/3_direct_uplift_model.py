from os.path import join as pjoin
from typing import List, Any

import numpy as np
import pandas as pd
import time
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgbm

from utils import uplift_score, DATA_PATH, SUBMISSIONS_PATH

import uplift.tree as uplift_tree

from sklearn.tree import DecisionTreeClassifier
from CTL.CTL import CausalTree

import multiprocessing


def parallel(func, n_processes: int, all_args: List[tuple]) -> List[Any]:
    if n_processes > 1:
        with multiprocessing.Pool(n_processes) as p:
            result = p.starmap(func, all_args)
    else:
        result = [func(*args) for args in all_args]
    return result


class UpliftRandomForestClassifier:
    def __init__(
            self,
            n_estimators: int = 100,
            # max_depth: int = 3,
            n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        # self.max_depth = max_depth
        self.n_jobs = n_jobs

        self.trees = []
        self.samples_col_indices = []

    @staticmethod
    def _fit_tree(
        X: np.ndarray,
        target: np.ndarray,
        treatment: np.ndarray,
        seed: int,
    ) -> CausalTree:
        # print('fit tree')
        ctl = CausalTree(min_size=100, max_depth=3, seed=seed)
        ctl.fit(X, target, treatment)
        return ctl

    def fit(self, X: np.ndarray, target: np.ndarray, treatment: np.ndarray):
        np.random.seed(1)

        n_rows, n_columns = X.shape
        n = n_rows
        m = 4
        all_row_indices = np.arange(n_rows)
        all_col_indices = np.arange(n_columns)

        all_args = []
        for i in range(self.n_estimators):
            row_indices = np.random.choice(all_row_indices, size=n,
                                           replace=True)
            col_indices = np.random.choice(all_col_indices, size=m,
                                           replace=False)
            sample = X[row_indices, :][:, col_indices]
            sample_target = target[row_indices]
            sample_treatment = treatment[row_indices]
            all_args.append((sample, sample_target, sample_treatment, i))
            self.samples_col_indices.append(col_indices)

        self.trees = parallel(self._fit_tree, self.n_jobs, all_args)

    @staticmethod
    def _predict_from_tree(ctl: CausalTree, X: np.ndarray) -> np.ndarray:
        prediction = ctl.predict(X)
        return prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        all_args = [
            (tree, X[:, col_indices])
            for tree, col_indices in zip(self.trees, self.samples_col_indices)
        ]
        predictions_all = parallel(self._predict_from_tree, self.n_jobs,
                                   all_args)
        predictions_mat = np.vstack(predictions_all)
        predictions = predictions_mat.mean(axis=0)
        return predictions

def uplift_fit_predict(X_train, treatment_train, target_train, X_test):
    """
    Реализация честного способа построения uplift-модели.

    """
    start_time = time.time()

    # model = uplift_tree.DecisionTreeClassifier(criterion='uplift_gini')
    # model.

    rf_clf = UpliftRandomForestClassifier(n_estimators=500, n_jobs=4)
    rf_clf.fit(X_train, target_train, treatment_train)
    predict_uplift = rf_clf.predict(X_test)

    # ctl = CausalTree(min_size=100, max_depth=4)
    # ctl.fit(X_train, target_train, treatment_train)
    # predict_uplift = ctl.predict(X_test)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Fitted and predicted in {elapsed:.1f} seconds')
    return predict_uplift


def round_df_column(df, column, round_to):
    df[column] = df[column] // round_to * round_to


def analyze_combs(df):
    combs = 1
    for col in df.columns:
        nunique = df[col].nunique()
        combs *= nunique
        # print(col, nunique)
    print('Combs: ', combs)
# Чтение данных

if __name__ == '__main__':
    df_clients = pd.read_csv(pjoin(DATA_PATH, 'clients.csv'), index_col='client_id', parse_dates=['first_issue_date', 'first_redeem_date'])
    df_train = pd.read_csv(pjoin(DATA_PATH, 'uplift_train.csv'), index_col='client_id')
    df_test = pd.read_csv(pjoin(DATA_PATH, 'uplift_test.csv'), index_col='client_id')

    # Извлечение признаков
    MIN_DATETIME = df_clients['first_issue_date'].min()
    SECONDS_IN_DAY = 60 * 60 * 24
    df_clients['first_issue_unixtime'] = (df_clients['first_issue_date'] - MIN_DATETIME).dt.total_seconds() // SECONDS_IN_DAY
    df_clients['first_redeem_unixtime'] = (df_clients['first_redeem_date'] - MIN_DATETIME).dt.total_seconds() // SECONDS_IN_DAY

    df_features = pd.DataFrame({
        'gender_M': (df_clients['gender'] == 'M').astype(int),
        'gender_F': (df_clients['gender'] == 'F').astype(int),
        'gender_U': (df_clients['gender'] == 'U').astype(int),
        'age': df_clients['age'],
        'first_issue_time': df_clients['first_issue_unixtime'],
        'first_redeem_time': df_clients['first_redeem_unixtime'],
        'issue_redeem_delay': df_clients['first_redeem_unixtime'] - df_clients['first_issue_unixtime'],
    }).fillna(0)




    # simplify
    df_features.loc[df_features['age'] < 0, 'age'] = -10
    df_features.loc[df_features['age'] > 80, 'age'] = -20
    round_df_column(df_features, 'age', 10)
    round_df_column(df_features, 'first_issue_time', 30)
    round_df_column(df_features, 'first_redeem_time', 30)
    round_df_column(df_features, 'issue_redeem_delay', 30)

    analyze_combs(df_features)
    # Оценка качества на валидации

    indices_train = df_train.index
    indices_test = df_test.index
    indices_learn, indices_valid = train_test_split(df_train.index, test_size=0.3, random_state=123)

    X_train = df_features.loc[indices_train, :].values
    treatment_train = df_train.loc[indices_train, 'treatment_flg'].values
    target_train = df_train.loc[indices_train, 'target'].values

    X_learn = df_features.loc[indices_learn, :].values
    treatment_learn = df_train.loc[indices_learn, 'treatment_flg'].values
    target_learn = df_train.loc[indices_learn, 'target'].values

    X_valid = df_features.loc[indices_valid, :].values
    treatment_valid = df_train.loc[indices_valid, 'treatment_flg'].values
    target_valid = df_train.loc[indices_valid, 'target'].values

    X_test = df_features.loc[indices_test, :].values


    valid_uplift = uplift_fit_predict(
        X_train=X_learn,
        treatment_train=treatment_learn,
        target_train=target_learn,
        X_test=X_valid,
    )
    valid_score = uplift_score(
        valid_uplift,
        treatment=treatment_valid,
        target=target_valid,
    )
    print('Validation score:', valid_score)


    # Подготовка предсказаний для тестовых клиентов
    # test_uplift = uplift_fit_predict(
    #     X_train=X_train,
    #     treatment_train=treatment_train,
    #     target_train=target_train,
    #     X_test=X_test,
    # )
    #
    # df_submission = pd.DataFrame({'uplift': test_uplift}, index=df_test.index)
    # df_submission.to_csv(pjoin(SUBMISSIONS_PATH, 'submission_direct_uplift_model.csv'))
