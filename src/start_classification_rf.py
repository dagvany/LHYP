#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import math
from pathlib import Path

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from models import Pathology

from utils import get_logger
from proc_utils import getConfiguration

logger = get_logger(__name__)


def getInputAndLabelSets(latentData, width, height):
    labelsTrainSet = []
    inputsTrainSet = []
    for d in latentData:
        if d[0] == Pathology.NORMAL:
            labelsTrainSet.append(0.0)
        else:
            labelsTrainSet.append(1.0)
        latentTensor = d[1].view(width * height)
        latentNumpy = latentTensor.detach().numpy()
        inputsTrainSet.append(latentNumpy)

    return inputsTrainSet, labelsTrainSet


config = getConfiguration("config_classification.json")
pathTrain = os.path.join(
    config["root"],
    config['latent_pickle'],
    config['latent_train_file'])
pathValidate = os.path.join(
    config["root"],
    config['latent_pickle'],
    config['latent_validate_file'])
pathModel = os.path.join(config["root"], config["pytorch_model_folder"])
timestr = time.strftime("%Y%m%d-%H%M%S")

# Create Classifier
regressor = RandomForestRegressor(n_estimators=20, random_state=0)

# TRAIN
with open(pathTrain, 'rb') as fp:
    latentTrainData = pickle.load(fp)

shape = latentTrainData[0][1].shape
width = shape[3]
height = shape[2]
random.shuffle(latentTrainData)
inputsTrainSet, labelsTrainSetValue = getInputAndLabelSets(
    latentTrainData, width, height)

regressor.fit(inputsTrainSet, labelsTrainSetValue)

# VALIDATE
with open(pathValidate, 'rb') as fp:
    latentValidateData = pickle.load(fp)
inputsValidateSet, labelsValidateSetValue = getInputAndLabelSets(
    latentValidateData, width, height)

outputs = regressor.predict(inputsValidateSet)
print('Mean Absolute Error:', metrics.mean_absolute_error(
    labelsValidateSetValue, outputs))
print('Mean Squared Error:', metrics.mean_squared_error(
    labelsValidateSetValue, outputs))
print('Root Mean Squared Error:', np.sqrt(
    metrics.mean_squared_error(labelsValidateSetValue, outputs)))

outputs = np.round(outputs)
tn, fp, fn, tp = metrics.confusion_matrix(
    labelsValidateSetValue, outputs).ravel()
cmMsg = "TP: {} FP: {} FN: {} TN: {}".format(tp, fp, fn, tn)
logger.info(cmMsg)
print(cmMsg)
accMsg = "Accuracy:", metrics.accuracy_score(labelsValidateSetValue, outputs)
logger.info(accMsg)
print(accMsg)
f1Msg = "F1:", metrics.f1_score(labelsValidateSetValue, outputs)
logger.info(f1Msg)
print(f1Msg)
recallMsg = "Recall:", metrics.recall_score(labelsValidateSetValue, outputs)
logger.info(recallMsg)
print(recallMsg)
