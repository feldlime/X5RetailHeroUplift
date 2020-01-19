import pandas
import datetime
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


def uplift_fit_predict(model, X_train, treatment_train, target_train, X_test):
    """
    Реализация простого способа построения uplift-модели.
    
    Обучаем два бинарных классификатора, которые оценивают вероятность target для клиента:
    1. с которым была произведена коммуникация (treatment=1)
    2. с которым не было коммуникации (treatment=0)
    
    В качестве оценки uplift для нового клиента берется разница оценок вероятностей:
    Predicted Uplift = P(target|treatment=1) - P(target|treatment=0)
    """
    X_treatment, y_treatment = X_train[treatment_train == 1, :], target_train[treatment_train == 1]
    X_control, y_control = X_train[treatment_train == 0, :], target_train[treatment_train == 0]
    model_treatment = clone(model).fit(X_treatment, y_treatment)
    model_control = clone(model).fit(X_control, y_control)
    predict_treatment = model_treatment.predict_proba(X_test)[:, 1]
    predict_control = model_control.predict_proba(X_test)[:, 1]
    predict_uplift = predict_treatment - predict_control
    return predict_uplift


def uplift_score(prediction, treatment, target, rate=0.3):
    """
    Подсчет Uplift Score
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score


# Чтение данных

df_clients = pandas.read_csv('data/clients.csv', index_col='client_id')
df_train = pandas.read_csv('data/uplift_train.csv', index_col='client_id')
df_test = pandas.read_csv('data/uplift_test.csv', index_col='client_id')

# Извлечение признаков

df_clients['first_issue_unixtime'] = pandas.to_datetime(df_clients['first_issue_date']).astype(int)/10**9
df_clients['first_redeem_unixtime'] = pandas.to_datetime(df_clients['first_redeem_date']).astype(int)/10**9
df_features = pandas.DataFrame({
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

df_submission = pandas.DataFrame({'uplift': test_uplift}, index=df_test.index)
df_submission.to_csv('submission.csv')
