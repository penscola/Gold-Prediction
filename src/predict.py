import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = '/media/penscola/Penscola@Tech/Projects/Gold-Prediction/model/Random-Forest-Regressor.pkl'

with open(model_file, 'rb') as f_in:
    scaler, model = pickle.load(f_in)

app = Flask('price')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = scaler.transform([customer])
    y_pred = model.predict(X)
    price = y_pred >= 0.5

    result = {
        'price_probability': float(y_pred),
        'price': bool(price)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)