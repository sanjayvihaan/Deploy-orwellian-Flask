from flask import Flask, request, jsonify
import joblib
import numpy as np
from features import FeatureExtraction

app = Flask(__name__)

gbc = joblib.load('gbc_final_model.pkl')


@app.route('/', methods=['GET'])
def hello():
    response = {"result": "hello world"}
    return jsonify(response), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data['url']

        # Extract features using FeatureExtraction class
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        # Make predictions
        y_pred = gbc.predict(x)[0]

        # Define response messages based on your model's output
        if y_pred == 1:
            response = {"result": "safe"}
        else:
            response = {"result": "phishing"}

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
