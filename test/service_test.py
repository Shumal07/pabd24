import time
import numpy as np
import requests
from dotenv import dotenv_values
import pandas as pd
from joblib import load

endpoint = 'http://127.0.0.1:5000/predict'

config = dotenv_values(".env")
HEADERS = {"Authorization": f"Bearer {config['APP_TOKEN']}"}
proxies = {
  "http": None,
  "https": None,
}

# Load the LabelEncoder
label_encoder = load('models/le_underground.joblib')

def encode_underground(value):
    try:
        return int(label_encoder.transform([value])[0])
    except ValueError:
        return int(label_encoder.transform(['unknown'])[0])

def preprocess_record(record):
    record['underground'] = encode_underground(record['underground'])
    # Convert all values to standard Python types
    record = {k: int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in record.items()}
    return record

def do_request(data: dict) -> tuple:
    t0 = time.time()
    resp = requests.post(
        endpoint,
        json=data,
        headers=HEADERS,
        proxies=proxies
    ).json()
    t = time.time() - t0
    return t, resp['price']

def test_100():
    df = pd.read_csv('data/proc/val.csv')
    prices = df['price']
    df = df.drop(['price'], axis=1)
    records = df.to_dict('records')
    delays, pred_prices = [], []
    for row in records:
        row = preprocess_record(row)
        t, price = do_request(row)
        delays.append(t)
        pred_prices.append(price)
    avg_delay = sum(delays) / len(delays)
    error = np.array(pred_prices) - prices.to_numpy()
    avg_error = np.mean(error)
    print(f'Avg delay: {avg_delay*1000} ms, avg error: {avg_error} RUB')

if __name__ == '__main__':
    test_100()
