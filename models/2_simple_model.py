import logging
from os.path import join as pjoin

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from models.metrics import uplift_metrics
from config import DATA_PATH, SUBMISSIONS_PATH

logfile_format = '[%(asctime)s] %(name)-10s %(levelname)-8s %(message)s'
logging.basicConfig(format=logfile_format, level=logging.DEBUG)


def uplift_fit_predict(X_train, treatment_train, target_train, X_test):
    """
    Реализация чуть более сложного способа построения uplift-модели.
    
    Обучаем бинарный классификатор с целевой переменной:
    Z = Y * W + (1 - Y) * (1 - W)
    где Y - target (купил / не купил),
    W - treatment (было воздействие / не было)

    Uplift считаем по формуле (Бабушкин 5:34):
    Predicted Uplift = 2 * P(Z=1) - 1
    """
    y = target_train
    w = treatment_train
    z = y * w + (1 - y) * (1 - w)

    model = GradientBoostingClassifier(random_state=1)
    model.fit(X_train, z)

    predict_z = model.predict_proba(X_test)[:, 1]
    predict_uplift = 2 * predict_z - 1
    return predict_uplift


def load_data():
    df_clients = pd.read_csv(
        pjoin(DATA_PATH, 'clients.csv'),
        index_col='client_id',
        parse_dates=['first_issue_date', 'first_redeem_date'],
    )
    df_train = pd.read_csv(
        pjoin(DATA_PATH, 'uplift_train.csv'),
        index_col='client_id',
    )
    df_test = pd.read_csv(
        pjoin(DATA_PATH, 'uplift_test.csv'),
        index_col='client_id',
    )
    return df_clients, df_train, df_test


def create_features(df_clients: pd.DataFrame):
    # Извлечение признаков
    min_datetime = df_clients['first_issue_date'].min()
    seconds_in_day = 60 * 60 * 24
    df_clients['first_issue_unixtime'] = (
        (df_clients['first_issue_date'] - min_datetime)
        .dt.total_seconds() /
        seconds_in_day
    )
    df_clients['first_redeem_unixtime'] = (
        (df_clients['first_redeem_date'] - min_datetime)
        .dt.total_seconds() /
        seconds_in_day
    )

    df_features = pd.DataFrame({
        'gender_M': (df_clients['gender'] == 'M').astype(int),
        'gender_F': (df_clients['gender'] == 'F').astype(int),
        'gender_U': (df_clients['gender'] == 'U').astype(int),
        'age': df_clients['age'],
        'first_issue_time': df_clients['first_issue_unixtime'],
        'first_redeem_time': df_clients['first_redeem_unixtime'],
        'issue_redeem_delay': df_clients['first_redeem_unixtime'] - df_clients[
            'first_issue_unixtime'],
    }).fillna(0)

    return df_features


def main():
    logging.info('Loading data...')
    clients, train, test = load_data()

    logging.info('Creating features...')
    features = create_features(df_clients=clients)

    logging.info('Preparing samples...')

    indices_train = train.index
    indices_test = test.index
    indices_learn, indices_valid = train_test_split(
        train.index,
        test_size=0.3,
        random_state=123,
    )

    X_train = features.loc[indices_train, :].values
    treatment_train = train.loc[indices_train, 'treatment_flg'].values
    target_train = train.loc[indices_train, 'target'].values

    X_learn = features.loc[indices_learn, :].values
    treatment_learn = train.loc[indices_learn, 'treatment_flg'].values
    target_learn = train.loc[indices_learn, 'target'].values

    X_valid = features.loc[indices_valid, :].values
    treatment_valid = train.loc[indices_valid, 'treatment_flg'].values
    target_valid = train.loc[indices_valid, 'target'].values

    X_test = features.loc[indices_test, :].values

    logging.info('Training and prediction for validation...')
    valid_uplift = uplift_fit_predict(
        X_train=X_learn,
        treatment_train=treatment_learn,
        target_train=target_learn,
        X_test=X_valid,
    )
    valid_score = uplift_metrics(
        valid_uplift,
        treatment=treatment_valid,
        target=target_valid,
    )
    logging.info(f'Validation score: {valid_score}')

    logging.info('Training and prediction for test...')
    test_uplift = uplift_fit_predict(
        X_train=X_train,
        treatment_train=treatment_train,
        target_train=target_train,
        X_test=X_test,
    )

    df_submission = pd.DataFrame({'uplift': test_uplift}, index=indices_test)
    df_submission.to_csv(pjoin(SUBMISSIONS_PATH, 'submission_test.csv'))
    logging.info('Submission is ready')


if __name__ == '__main__':
    main()
