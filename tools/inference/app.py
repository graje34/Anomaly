from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
import cv2
import time
from lightning_inference import *



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


weight_file="/home/grajebhosle/Documents/IPML/Projects/Anamoly/anomalib_main/tools/inference/Weight/model.ckpt"

def model_predict(file_path,threshold):
    img = file_path
    confidence=infer(img,weight_file)
    pred_label= confidence[0]['pred_labels'][0].item()
    confidence=confidence[0]['pred_scores'][0].item()
    print("prediction label is",pred_label)
    print("confidence is",confidence)
    if confidence == 0.0:
        confidence = 1
        confidence=confidence*100

    else:
        confidence = confidence * 100
    
    if pred_label==False:
        if confidence > threshold:
            pred = "OK"
        elif confidence < threshold:
            pred = "Anomaly"
    elif pred_label==True:
        pred = "Anomaly"
    confidence=str(confidence)
    return pred,confidence


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
            # check if the file is an image
        if f.filename.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg']:
            result=str("Invalid file format. Please provide an image in PNG or JPEG format")
            return result
            
        else:    
        
            # Check if the threshold parameter is present in the request
            if 'threshold' in request.form:
                threshold = float(request.form['threshold'])
            else:
                threshold = 20

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds,confidence= model_predict(file_path,threshold)
            result = "Prediction: {}, Confidence: {}".format(preds, confidence)


            return result
    return None
    

    


if __name__ == '__main__':
    app.run(debug=True)
