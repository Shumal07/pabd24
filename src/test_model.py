import argparse
import logging
import pandas as pd
from sklearn.metrics import mean_absolute_error
from joblib import load

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/train_and_val_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

VAL_DATA = 'data/proc/val.csv'
MODEL_PATH = 'models/random_forest_v2.joblib'

def main(args):
    df_val = pd.read_csv(VAL_DATA)
    x_val = df_val.drop(columns=["price", "url_id"])
    y_val = df_val['price']

    model = load(args.model)
    logger.info(f'Loaded model {args.model}')

    y_pred = model.predict(x_val)
    mae = mean_absolute_error(y_pred, y_val)
    logger.info(f'MAE = {mae:.0f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Model load path', default=MODEL_PATH)
    args = parser.parse_args()
    main(args)
