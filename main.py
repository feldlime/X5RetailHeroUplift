import logging
from datetime import timedelta

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
from utils import RANDOM_STATE, save_submission

log_format = '[%(asctime)s] %(name)-15s %(levelname)-8s %(message)s'
logging.basicConfig(
    format=log_format,
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def main():

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
    # TODO: normal fill na
    features.fillna(-2)
    features.set_index('client_id', inplace=True)
    logger.info('Features are ready')

    logger.info('Preparing data sets...')
    train = load_train()
    test = load_test()
    indices_train = train.index
    indices_test = test.index

    X_train = features.loc[indices_train, :].values
    treatment_train = train.loc[indices_train, 'treatment_flg'].values
    target_train = train.loc[indices_train, 'target'].values

    X_test = features.loc[indices_test, :].values

    # TODO: Instead of this do cross validation and grid search
    indices_learn, indices_valid = train_test_split(
        train.index,
        test_size=0.3,
        random_state=RANDOM_STATE,
    )

    X_learn = features.loc[indices_learn, :].values
    treatment_learn = train.loc[indices_learn, 'treatment_flg'].values
    target_learn = train.loc[indices_learn, 'target'].values

    X_valid = features.loc[indices_valid, :].values
    treatment_valid = train.loc[indices_valid, 'treatment_flg'].values
    target_valid = train.loc[indices_valid, 'target'].values
    logger.info('Data sets prepared')

    clf = LGBMClassifier(
        boosting_type='rf',
        n_estimators=100,
        num_leaves=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-3,
        bagging_freq=1,
        bagging_fraction=0.5,
    )

    logger.info('Build model for learn data set...')
    clf = uplift_fit(clf, X_learn, treatment_learn, target_learn)
    learn_pred = uplift_predict(clf, X_learn)
    learn_scores = uplift_metrics(learn_pred, treatment_learn, target_learn)
    valid_pred = uplift_predict(clf, X_valid)
    valid_scores = uplift_metrics(valid_pred, treatment_valid, target_valid)
    logger.info(f'Learn scores: {learn_scores}')
    logger.info(f'Valid scores: {valid_scores}')

    logging.info('Build model for full train data set...')
    clf = uplift_fit(clf, X_train, treatment_train, target_train)
    train_pred = uplift_predict(clf, X_train)
    train_scores = uplift_metrics(train_pred, treatment_train, target_train)
    logger.info(f'Train scores: {train_scores}')
    test_pred = uplift_predict(clf, X_test)

    logger.info('Saving submission...')
    save_submission(indices_test, test_pred, 'submission_.csv')
    logger.info('Submission is ready')

if __name__ == '__main__':
    main()



