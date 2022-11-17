# Install libraries
from inspect import trace
import sys
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/prediction/hairloss', methods=['POST'])
# define function
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            predict = list(model.predict(query))
            print(predict)
            return jsonify({'prediction': str(predict)})

        except:
            return jsonify({'trace': traceback.format_exc()})

    else:
        print("Model is not good")
        return ('Model is not good')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 4002

        model = joblib.load("hairloss.pkl")
        print("Model Loaded")
        model_columns = joblib.load("hairloss_columns.pkl")
        # Load "rnd_columns.pkl"
        print("Model columns loaded")

        # app.run(host="203.247.240.226", port=port, debug=True)
        app.run(host="203.247.240.226",port=port, debug=True)