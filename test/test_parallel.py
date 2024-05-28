import time
from multiprocessing import Pool
import requests
from dotenv import dotenv_values

config = dotenv_values(".env")
endpoint = 'http://192.144.14.183:8000/'
HEADERS = {"Authorization": f"Bearer {config['APP_TOKEN']}"}


def do_request(area: int) -> str:
    data = {'area': area}
    t0 = time.time()
    resp = requests.post(
        endpoint,
        json=data,
        headers=HEADERS
    )
    t = time.time() - t0
    return f'Waited {t:0.2f} sec ' + resp.text


def test_10():
    with Pool(10) as p:
        print(*p.map(do_request, range(10, 110, 10)))


if __name__ == '__main__':
    test_10()