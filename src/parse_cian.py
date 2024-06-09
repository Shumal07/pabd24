"""  Parse data from cian.ru
https://github.com/lenarsaitov/cianparser
"""
import datetime

import cianparser
import pandas as pd

moscow_parser = cianparser.CianParser(location="Москва")


def main(n_rooms: int, start_page: int, end_page: int):
    """
    Function docstring
    """
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f'../data/raw/{n_rooms}_pages_{start_page}_to_{end_page}_{t}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        additional_settings={
            "start_page": start_page,
            "end_page": end_page,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(csv_path,
              encoding='utf-8',
              index=False)


if __name__ == '__main__':
    for i in range(1, 4):
        for start in range(1, 151, 50):
            end = start + 49
            main(i, start, end)