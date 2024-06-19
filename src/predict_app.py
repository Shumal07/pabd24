"""House price prediction service"""
from dotenv import dotenv_values
from flask import Flask, request
from flask_cors import CORS
from joblib import load
from flask_httpauth import HTTPTokenAuth

#5000 - flask
#8000 - gunicorn

app = Flask(__name__)
CORS(app)

config = dotenv_values(".env")
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    config['APP_TOKEN']: "user1",
}
model = load('models/catboost.joblib')


@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]

def predict(in_data: dict) -> int:
    """ Predict house price from input data parameters.
    :param in_data: house parameters.
    :raise Error: If something goes wrong.
    :return: House price, RUB.
    :rtype: int
    """
    rooms_count = int(in_data['rooms_count'])
    author_type = str(in_data['author_type'])
    floor = int(in_data['floor'])
    street = str(in_data['street'])
    underground = str(in_data['underground'])
    floors_count = int(in_data['floors_count'])
    district = str(in_data['district'])
    total_meters = float(in_data['total_meters'])

    price = model.predict([[rooms_count, author_type, floor, street, underground, floors_count,district, total_meters]])[0]

    return int(price.squeeze())


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