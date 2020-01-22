import pandas as pd


def make_client_features(clients: pd.DataFrame) -> pd.DataFrame:
    """No id in index"""

    min_datetime = clients['first_issue_date'].min()
    seconds_in_day = 60 * 60 * 24
    first_issue_time = (
            (clients['first_issue_date'] - min_datetime)
            .dt.total_seconds() /
            seconds_in_day
    ).values
    first_redeem_time = (
            (clients['first_redeem_date'] - min_datetime)
            .dt.total_seconds() /
            seconds_in_day
    ).values

    age = clients['age'].values
    age[age < 0] = -2
    age[age > 100] = -3

    gender = clients['gender'].values

    features = pd.DataFrame({
        'client_id': clients['client_id'].values,
        'gender_M': (gender == 'M').astype(int),
        'gender_F': (gender == 'F').astype(int),
        'gender_U': (gender == 'U').astype(int),
        'age': age,
        'first_issue_time': first_issue_time,
        'first_redeem_time': first_redeem_time,
        'issue_redeem_delay': first_redeem_time - first_issue_time,
    })

    features = features.fillna(-1)

    return features
