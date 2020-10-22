# EY Cyber Threat Science - DGA Classifier
# This is Flask API that classifies domains as malicious or benign.
# Author: Alistair Gillespie, Machine Learning Engineer
# -------------------------------------------------------------------------------
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from importlib import import_module, reload
import torch
import math
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from flask import send_from_directory, render_template
from flask import Flask, jsonify, request
from flask_bootstrap import Bootstrap
from torch.utils.data import DataLoader

# For Flask in a sub-directory and not installed as a package
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from dga.models.dga_classifier import DGAClassifier
from dga.datasets.domain_dataset import DomainDataset

# -------------------------------------------------------------------------------
# python entry point to run the flask app
app = Flask(__name__)

Bootstrap(app)

model_dir = 'models/'
model_info = {}
model_info_path = os.path.join(model_dir, '20201014_08-09-16_dga_model_info.pth')

with open(model_info_path, 'rb') as f:
    model_info = torch.load(f)

print("model_info: {}".format(model_info))

# Determine the device and construct the model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DGAClassifier(input_features=model_info['input_features'],
                      hidden_dim=model_info['hidden_dim'],
                      n_layers=model_info['n_layers'],
                      output_dim=model_info['output_dim'],
                      embedding_dim=model_info['embedding_dim'],
                      batch_size=model_info['batch_size'])

# Load the stored model parameters.
model_path = os.path.join(model_dir, '20201014_08-09-16_dga_model.pth')
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f))

# set to eval mode, could use no_grad
model.to(device).eval()

print("Done loading model.")


# -------------------------------------------------------------------------------
# GLOBAL variables 
# -------------------------------------------------------------------------------
app.Model = {}

# -------------------------------------------------------------------------------
# HELPER functions
# -------------------------------------------------------------------------------
# helper function: clean param
def get_clean_param(p):
    return p.lstrip("\"").rstrip("\"")


def get_prediction(df):
    predict_dl = _get_predict_loader(int(model_info['batch_size']), df)
    classes = {0: 'Benign', 1: 'DGA'}
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_num, (x_padded,  x_lens) in enumerate(predict_dl):
            output = model(x_padded, x_lens)
            y_hat = torch.round(output.data)
            predictions += [classes[int(key)] for key in y_hat.flatten().numpy()]

    return predictions


def _get_predict_loader(batch_size, df):
    print("Getting test and train data loaders.")
    dataset = DomainDataset(df, train=False)
    predict_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_pred)
    return predict_dl


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, yy, x_lens, y_lens


def pad_collate_pred(batch):
    x_lens = [len(x) for x in batch]
    xx_pad = pad_sequence(batch, batch_first=True, padding_value=0)
    return xx_pad, x_lens


def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())


# -------------------------------------------------------------------------------
# MAIN FLASK APP endpoint definitions
# -------------------------------------------------------------------------------
# set an icon if requested
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/append', methods=['POST'])
def append():

    # prepare a return object
    response = {}
    response["status"] = "error"
    response["message"] = "/predict: ERROR: "

    # 1. validate input POST data
    try:
        dp = request.get_json(force=True)
        # print("Received:", dp)
        app.Model["data"] = list(dp["data"])
        print("/apply: raw data size: ", len(app.Model["data"]))
    except Exception as e:
        response["message"] += 'unable to parse json from POST data. Provide a JSON object with structure { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}. Ended with exception: ' + str(e)
        return json.dumps(response)

    # 2. convert to dataframe
    try:
        # TODO check with compression option and chunked mode
        # FIX for tensorflow-gpu image does not have compat StringIO
        try:
            from pd.compat import StringIO
        except ImportError:
            from io import StringIO
        app.Model["df"] = pd.DataFrame(app.Model["data"])
        print("/append: dataframe shape: ", str(app.Model["df"].shape))
        # free memory from raw data
        
    except Exception as e:
        response["message"] += 'unable to convert raw data to pandas dataframe. Ended with exception: ' + str(e)
        return json.dumps(response)

    try:
        # print("Predict:", get_prediction(app.Model['df']))
        # print("Entropy:", app.Model['df'].iloc[:,0].apply(entropy).values)
        # print("Predict:", get_prediction(app.Model['df']))
        # print("Entropy:", app.Model['df'].iloc[:,0].apply(entropy).values)
        preds = get_prediction(app.Model["df"])
        entrs = app.Model["df"].iloc[:, 0].apply(entropy).values.tolist()
        # print(app.Model["data"], preds, entrs)
        results = zip(app.Model["data"], preds, entrs)
        del(app.Model["data"])
        # print(f"Results {list(results)}")
        df_result = pd.DataFrame(list(results), columns=["domain", "prediction", "entropy"])
        print("/predict: returned result dataframe with shape " + str(df_result.shape) + "")
    except Exception as e:
        response["message"] += 'unable to apply model. Ended with exception: ' + str(e)
        return json.dumps(response)

    df_result.to_csv('data/logs.csv', mode='a', header=False)

    # end with a successful response
    response["status"] = "success"
    response["message"] = "/append done successfully"
    return json.dumps(response)


@app.route('/predict', methods=['POST'])
def predict():

    # prepare a return object
    response = {}
    response["status"] = "error"
    response["message"] = "/predict: ERROR: "

    # 1. validate input POST data
    try:
        dp = request.get_json(force=True)
        # print("Received:", dp)
        app.Model["data"] = list(dp["data"])
        print("/apply: raw data size: ", len(app.Model["data"]))
    except Exception as e:
        response["message"] += 'unable to parse json from POST data. Provide a JSON object with structure { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}. Ended with exception: ' + str(e)
        return json.dumps(response)

    # 2. convert to dataframe
    try:
        # TODO check with compression option and chunked mode
        # FIX for tensorflow-gpu image does not have compat StringIO
        try:
            from pd.compat import StringIO
        except ImportError:
            from io import StringIO
        app.Model["df"] = pd.DataFrame(app.Model["data"])
        print("/predict: dataframe shape: ", str(app.Model["df"].shape))
        # free memory from raw data
        # app.Model["data"]
    except Exception as e:
        response["message"] += 'unable to convert raw data to pandas dataframe. Ended with exception: ' + str(e)
        return json.dumps(response)

    try:
        # print("Predict:", get_prediction(app.Model['df']))
        # print("Entropy:", app.Model['df'].iloc[:,0].apply(entropy).values)
        preds = get_prediction(app.Model["df"])
        entrs = app.Model["df"].iloc[:, 0].apply(entropy).values.tolist()
        # print(app.Model["data"], preds, entrs)
        results = zip(app.Model["data"], preds, entrs)
        # print(f"Results {list(results)}")
        df_result = pd.DataFrame(list(results), columns=["domain", "prediction", "entropy"])
        print("/predict: returned result dataframe with shape " + str(df_result.shape) + "")
    except Exception as e:
        response["message"] += 'unable to apply model. Ended with exception: ' + str(e)
        return json.dumps(response)

    response["results"] = df_result.to_csv(index=False)

    # end with a successful response
    response["status"] = "success"
    response["message"] = "/predict done successfully"
    return json.dumps(response)


@app.route("/")
def show_tables():
    data = pd.read_csv('data/logs.csv')
    # data.set_index(['domain'], inplace=True)
    return render_template(
        'view.html',
        tables=[data.to_html(classes='domains')],
        titles=['DNS Logs']
    )

# TODO: Look at waitress
# # -------------------------------------------------------------------------------
# # python entry point to run the flask app
# if __name__ == "__main__":
#     serve(app, host="0.0.0.0", port=5000)
#     FLASK_ENV=development flask run
#     app.run(ssl_context='adhoc')