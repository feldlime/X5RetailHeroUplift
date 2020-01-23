import logging
import pickle
from datetime import timedelta

import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

from features.client import make_client_features
from features.product import make_product_features
from features.purchase import make_purchase_features
from load_and_prepare import (
    prepare_clients,
    prepare_products,
    prepare_purchases,
    load_train,
    load_test,
)
from models.fit_predict import uplift_fit, uplift_predict
from models.metrics import uplift_metrics
from models.utils import make_z
from utils import save_submission
from config import RANDOM_STATE, N_ESTIMATORS

log_format = '[%(asctime)s] %(name)-25s %(levelname)-8s %(message)s'
logging.basicConfig(
    format=log_format,
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def prepare_features() -> pd.DataFrame:
    logger.info('Loading data...')
    clients, client_encoder = prepare_clients()
    products, product_encoder = prepare_products()
    purchases = prepare_purchases(client_encoder, product_encoder)
    del product_encoder
    logger.info('Data is loaded')

    logger.info('Preparing features...')
    purchase_features = make_purchase_features(purchases)

    # Last month purchases
    max_datetime = purchases['datetime'].max()
    cutoff = max_datetime - timedelta(days=30)
    purchases_lm = purchases[purchases['datetime'] >= cutoff]
    purchase_lm_features = make_purchase_features(purchases_lm)
    del purchases_lm

    purchases_ids = purchases.reindex(columns=['client_id', 'product_id'])
    del purchases
    product_features = make_product_features(products, purchases_ids)
    del purchases_ids

    client_features = make_client_features(clients)

    logger.info('Combining features...')
    features = (
        client_features
            .merge(purchase_features, on='client_id', how='left')
            .merge(purchase_lm_features, on='client_id', how='left')
            .merge(product_features, on='client_id', how='left')
    )
    del client_features
    del purchase_features
    del purchase_lm_features
    del product_features

    # TODO: normal fill na
    features.fillna(-2, inplace=True)

    features['client_id'] = client_encoder \
        .inverse_transform(features['client_id'])
    del client_encoder

    logger.info('Features are ready')

    return features

def main():
    # features = prepare_features()
    #
    # logger.info('Saving features...')
    # with open('features.pkl', 'wb') as f:
    #     pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    # logger.info('Features are saved')
    #
    logger.info('Loading features...')
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    logger.info('Features are loaded')

    logger.info('Preparing data sets...')
    features.set_index('client_id', inplace=True)

    train = load_train()
    test = load_test()
    indices_train = train.index
    indices_test = test.index

    X_train = features.loc[indices_train, :]
    treatment_train = train.loc[indices_train, 'treatment_flg'].values
    target_train = train.loc[indices_train, 'target'].values
    # y_valid = make_z(treatment_train, target_train)

    X_test = features.loc[indices_test, :]

    # TODO: Instead of this do cross validation and grid search
    indices_learn, indices_valid = train_test_split(
        train.index,
        test_size=0.3,
        random_state=RANDOM_STATE,
    )

    X_learn = features.loc[indices_learn, :]
    treatment_learn = train.loc[indices_learn, 'treatment_flg'].values
    target_learn = train.loc[indices_learn, 'target'].values
    # y_learn = make_z(treatment_learn, target_learn)

    X_valid = features.loc[indices_valid, :]
    treatment_valid = train.loc[indices_valid, 'treatment_flg'].values
    target_valid = train.loc[indices_valid, 'target'].values
    # y_valid = make_z(treatment_valid, target_valid)
    logger.info('Data sets prepared')

    clf_ = LGBMClassifier(
        boosting_type='rf',
        n_estimators=500,
        num_leaves=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-3,
        bagging_freq=1,
        bagging_fraction=0.5,
    )

    logger.info('Build model for learn data set...')
    clf = uplift_fit(clf_, X_learn, treatment_learn, target_learn)
    learn_pred = uplift_predict(clf, X_learn)
    learn_scores = uplift_metrics(learn_pred, treatment_learn, target_learn)
    valid_pred = uplift_predict(clf, X_valid)
    valid_scores = uplift_metrics(valid_pred, treatment_valid, target_valid)
    logger.info(f'Learn scores: {learn_scores}')
    logger.info(f'Valid scores: {valid_scores}')

    logging.info('Build model for full train data set...')
    clf = uplift_fit(clf_, X_train, treatment_train, target_train)
    train_pred = uplift_predict(clf, X_train)
    train_scores = uplift_metrics(train_pred, treatment_train, target_train)
    logger.info(f'Train scores: {train_scores}')
    test_pred = uplift_predict(clf, X_test)

    logger.info('Saving submission...')
    save_submission(indices_test, test_pred, 'submission_product_features_lgbm.csv')
    logger.info('Submission is ready')

if __name__ == '__main__':
    main()



