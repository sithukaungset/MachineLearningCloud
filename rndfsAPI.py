# Install libraries
from inspect import trace
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

application = Flask(__name__)


@application.route('/prediction', methods=['POST'])
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
