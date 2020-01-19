from os.path import join as pjoin

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from utils import uplift_score, DATA_PATH, SUBMISSIONS_PATH


def uplift_fit_predict(model, X_train, treatment_train, target_train, X_test):
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
    model = clone(model).fit(X_train, z)
    predict_z = model.predict_proba(X_test)[:, 1]
    predict_uplift = 2 * predict_z - 1
    return predict_uplift


# Чтение данных

df_clients = pd.read_csv(pjoin(DATA_PATH, 'clients.csv'), index_col='client_id', parse_dates=['first_issue_date', 'first_redeem_date'])
df_train = pd.read_csv(pjoin(DATA_PATH, 'uplift_train.csv'), index_col='client_id')
df_test = pd.read_csv(pjoin(DATA_PATH, 'uplift_test.csv'), index_col='client_id')

# Извлечение признаков
MIN_DATETIME = df_clients['first_issue_date'].min()


df_clients['first_issue_unixtime'] = (df_clients['first_issue_date'] - MIN_DATETIME).dt.total_seconds()/10**9
df_clients['first_redeem_unixtime'] = (df_clients['first_redeem_date'] - MIN_DATETIME).dt.total_seconds()/10**9

df_features = pd.DataFrame({
    'gender_M': (df_clients['gender'] == 'M').astype(int),
    'gender_F': (df_clients['gender'] == 'F').astype(int),
    'gender_U': (df_clients['gender'] == 'U').astype(int),
    'age': df_clients['age'],
    'first_issue_time': df_clients['first_issue_unixtime'],
    'first_redeem_time': df_clients['first_redeem_unixtime'],
    'issue_redeem_delay': df_clients['first_redeem_unixtime'] - df_clients['first_issue_unixtime'],
}).fillna(0)


# Оценка качества на валидации

indices_train = df_train.index
indices_test = df_test.index
indices_learn, indices_valid = train_test_split(df_train.index, test_size=0.3, random_state=123)

valid_uplift = uplift_fit_predict(
    model=GradientBoostingClassifier(),
    X_train=df_features.loc[indices_learn, :].fillna(0).values,
    treatment_train=df_train.loc[indices_learn, 'treatment_flg'].values,
    target_train=df_train.loc[indices_learn, 'target'].values,
    X_test=df_features.loc[indices_valid, :].fillna(0).values,
)
valid_score = uplift_score(
    valid_uplift,
    treatment=df_train.loc[indices_valid, 'treatment_flg'].values,
    target=df_train.loc[indices_valid, 'target'].values,
)
print('Validation score:', valid_score)


# Подготовка предсказаний для тестовых клиентов

test_uplift = uplift_fit_predict(
    model=GradientBoostingClassifier(),
    X_train=df_features.loc[indices_train, :].fillna(0).values,
    treatment_train=df_train.loc[indices_train, 'treatment_flg'].values,
    target_train=df_train.loc[indices_train, 'target'].values,
    X_test=df_features.loc[indices_test, :].fillna(0).values,
)

df_submission = pd.DataFrame({'uplift': test_uplift}, index=df_test.index)
df_submission.to_csv(pjoin(SUBMISSIONS_PATH, 'submission_simple_model.csv'))
