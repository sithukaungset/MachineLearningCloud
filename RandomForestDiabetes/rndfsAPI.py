# Install libraries
from inspect import trace
import sys
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/prediction/diabetes', methods=['POST'])
# define function
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=rnd_columns, fill_value=0)

            predict = list(lr.predict(query))

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
        port = 3002

        lr = joblib.load("randomfs.pkl")
        print("Model Loaded")
        rnd_columns = joblib.load("rnd_columns.pkl")
        # Load "rnd_columns.pkl"
        print("Model columns loaded")

        app.run(host="203.247.240.226",port=port, debug=True)
