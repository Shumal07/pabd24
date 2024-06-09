import argparse
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import dump

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/preprocess_data.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

IN_FILES = [
    'data/raw/1_pages_1_to_50_2024-06-09_15-31.csv',
    'data/raw/1_pages_51_to_100_2024-06-09_15-39.csv',
    'data/raw/1_pages_101_to_150_2024-06-09_15-59.csv',
    'data/raw/2_pages_1_to_50_2024-06-09_16-21.csv',
    'data/raw/2_pages_51_to_100_2024-06-09_16-29.csv',
    'data/raw/2_pages_101_to_150_2024-06-09_16-49.csv',
    'data/raw/3_pages_1_to_50_2024-06-09_17-10.csv',
    'data/raw/3_pages_51_to_100_2024-06-09_17-18.csv',
    'data/raw/3_pages_101_to_150_2024-06-09_17-39.csv'
]

OUT_TRAIN = 'data/proc/train.csv'
OUT_VAL = 'data/proc/val.csv'
TRAIN_SIZE = 0.9

def main(args):
    main_dataframe = pd.read_csv(args.input[0], delimiter=',')
    for i in range(1, len(args.input)):
        data = pd.read_csv(args.input[i], delimiter=',')
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
    new_dataframe = main_dataframe[['url_id', 'floor', 'floors_count', 'rooms_count', 'total_meters', 'district', 'street', 'underground', 'residential_complex', 'price']].set_index('url_id')

    new_df = new_dataframe[new_dataframe['price'] < 30_000_000].sample(frac=1)

    # Добавляем новые признаки
    new_df['first_floor'] = new_df['floor'] == 1
    new_df['last_floor'] = new_df['floor'] == new_df['floors_count']
    new_df['price_per_sqm'] = new_df['price'] / new_df['total_meters']

    # Обработка категориальных признаков
    le_district = LabelEncoder()
    le_street = LabelEncoder()
    le_underground = LabelEncoder()
    le_residential_complex = LabelEncoder()

    new_df['district'] = le_district.fit_transform(new_df['district'].fillna('unknown'))
    new_df['street'] = le_street.fit_transform(new_df['street'].fillna('unknown'))
    new_df['underground'] = le_underground.fit_transform(new_df['underground'].fillna('unknown'))
    new_df['residential_complex'] = le_residential_complex.fit_transform(new_df['residential_complex'].fillna('unknown'))

    # Сохранение LabelEncoders
    dump(le_district, 'models/le_district.joblib')
    dump(le_street, 'models/le_street.joblib')
    dump(le_underground, 'models/le_underground.joblib')
    dump(le_residential_complex, 'models/le_residential_complex.joblib')

    df = new_df[['floor', 'floors_count', 'rooms_count', 'total_meters', 'first_floor', 'last_floor', 'price_per_sqm', 'district', 'street', 'underground', 'residential_complex', 'price']]

    border = int(args.split * len(new_df))
    train_df, val_df = df.iloc[:border], df.iloc[border:]
    train_df.to_csv(OUT_TRAIN)
    val_df.to_csv(OUT_VAL)
    logger.info(f'Write {args.input} to train.csv and val.csv. Train set size: {len(train_df)}, Validation set size: {len(val_df)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=float, help='Split test size', default=TRAIN_SIZE)
    parser.add_argument('-i', '--input', nargs='+', help='List of input files', default=IN_FILES)
    args = parser.parse_args()
    main(args)
