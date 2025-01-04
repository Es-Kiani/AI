from flask import Flask, request, jsonify
from prediction import *

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_prediction():
    try:
        mint = request.args.get('mint', default=30, type=int)
        prediction = get_model_prediction(mint)
        return jsonify({"signal": prediction}), 200
    except Exception as e:
        return jsonify({"err": str(e)}), 500

if __name__ == '__main__':
    app.run(port=1313)
