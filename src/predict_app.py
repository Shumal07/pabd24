from flask import Flask, request
from flask_httpauth import HTTPTokenAuth
from flask_cors import CORS
from joblib import load
from dotenv import dotenv_values

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'models/random_forest_v2.joblib'
ENCODER_PATH = 'models/le_underground.joblib'
config = dotenv_values(".env")
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    config["APP_TOKEN"]: "user1",
}

@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]

label_encoder = load(ENCODER_PATH)

def encode_underground(value):
    try:
        return label_encoder.transform([value])[0]
    except ValueError:
        return label_encoder.transform(['unknown'])[0]

def predict(in_data: dict) -> int:
    total_meters = float(in_data['total_meters'])
    floor = int(in_data['floor'])
    floors_count = int(in_data['floors_count'])
    rooms_count = int(in_data['rooms_count'])
    district = int(in_data['district'])
    street = int(in_data['street'])
    underground = encode_underground(in_data['underground'])
    residential_complex = int(in_data['residential_complex'])
    first_floor = floor == 1
    last_floor = floor == floors_count
    price_per_sqm = float(in_data['price_per_sqm'])

    input_features = [
        floor,
        floors_count,
        rooms_count,
        total_meters,
        first_floor,
        last_floor,
        price_per_sqm,
        district,
        street,
        underground,
        residential_complex
    ]

    model = load(MODEL_PATH)
    res = model.predict([input_features])
    return int(res)

@app.route("/")
def home():
    return '<h1>Housing price service.</h1> Use /predict endpoint'

@app.route("/predict", methods=['POST'])
@auth.login_required
def predict_web_serve():
    in_data = request.get_json()
    price = predict(in_data)
    return {'price': price}

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
