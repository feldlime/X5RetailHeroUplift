from os.path import join as pjoin
from typing import List, Any

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split


from utils import uplift_score, DATA_PATH, SUBMISSIONS_PATH


from uplift.ensemble import RandomForestClassifier  as URFC

def uplift_fit_predict(X_train, treatment_train, target_train, X_test):
    """
    Реализация честного способа построения uplift-модели.

    """
    start_time = time.time()

    rf_clf = URFC(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=200,
        criterion='uplift_gini',
        n_jobs=5,
        random_state=1
    )
    rf_clf.fit(X_train, target_train, treatment_train)
    predict_uplift = rf_clf.predict_uplift(X_test)


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
    df_clients['first_issue_unixtime'] = (df_clients['first_issue_date'] - MIN_DATETIME).dt.total_seconds() / SECONDS_IN_DAY
    df_clients['first_redeem_unixtime'] = (df_clients['first_redeem_date'] - MIN_DATETIME).dt.total_seconds() / SECONDS_IN_DAY

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
    # df_features.loc[df_features['age'] < 0, 'age'] = -10
    # df_features.loc[df_features['age'] > 80, 'age'] = -20
    # round_df_column(df_features, 'age', 10)
    # round_df_column(df_features, 'first_issue_time', 30)
    # round_df_column(df_features, 'first_redeem_time', 30)
    # round_df_column(df_features, 'issue_redeem_delay', 30)
    #
    # analyze_combs(df_features)
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
