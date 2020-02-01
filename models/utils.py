import pandas as pd

def make_z(treatment, target):
    y = target
    w = treatment
    z = y * w + (1 - y) * (1 - w)
    return z


def calc_uplift(prediction):
    uplift = 2 * prediction - 1
    return uplift


def get_feature_importances(est, columns):
    return pd.DataFrame({
        'column': columns,
        'importance': est.feature_importances_,
    }).sort_values('importance', ascending=False)
