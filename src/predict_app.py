"""House price prediction service"""

from flask import Flask, request
from flask_httpauth import HTTPTokenAuth
from flask_cors import CORS
from joblib import load
from dotenv import dotenv_values
import time
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'models/linear_regression_v01.joblib'
config = dotenv_values(".env")
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    config["APP_TOKEN"]: "user1",
}

@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]

def predict_cpu_bounded(area, n=10000000000000000000000000000000000000):
    """Emulate single thread computation"""
    avg_price = sum([x for x in range(n)]) / n
    return int(area * avg_price)


def predict(in_data: dict) -> int:
    """ Predict house price from input data parameters.
    :param in_data: house parameters.
    :raise Error: If something goes wrong.
    :return: House price, RUB.
    :rtype: int
    """
    area = float(in_data['area'])
    #model = load(MODEL_PATH)
    price = predict_cpu_bounded(area)
    return price


@app.route("/")
def home():
    return '<h1>Housing price service.</h1> Use /predict endpoint'


@app.route("/predict", methods=['POST'])
@auth.login_required
def predict_web_serve():
    """Dummy service"""
    in_data = request.get_json()
    price = predict(in_data)
    return {'price': price}


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)