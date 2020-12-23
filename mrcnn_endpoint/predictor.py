# -*- coding: utf-8 -*-
import sys
import os
import argparse
import logging
import warnings
import io
import json
import boto3

import warnings
import numpy as np

from PIL import Image
import itertools
import cv2
import skimage.io

warnings.filterwarnings("ignore", category=FutureWarning)

import sys

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import torchvision
import PIL
import codecs

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import flask

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    # parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)

    data = flask.request.data.decode('utf-8')
    data = json.loads(data)

    bucket = data['bucket']
    image_uri = data['image_uri']

    download_file_name = image_uri.split('/')[-1]
    print ("<<<<download_file_name ", download_file_name)
    #download_file_name = './test.jpg'
    s3_client.download_file(bucket, image_uri, download_file_name)
    print('Download finished!')
    # inference and send result to RDS and SQS

    print('Start to inference:')

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # set it to evaluation mode, as the model behaves differently
    # during training and during evaluation
    model.eval()

    image = PIL.Image.open(download_file_name)
    image_tensor = torchvision.transforms.functional.to_tensor(image)

    # pass a list of (potentially different sized) tensors
    # to the model, in 0-1 range. The model will take care of
    # batching them together and normalizing
    output = model([image_tensor])

    if 3 in output[0]['labels'][output[0]['scores'] > 0.5]:
        label = 'withcar'
    else:
        label = 'withoutcar'

    # output is a list of dict, containing the postprocessed predictions
    result = {
        'label': label
    }

    print ('<<<< result: ', result)

    inference_result = {
        'result': result
    }
    _payload = json.dumps(inference_result, ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')