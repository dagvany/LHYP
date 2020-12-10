#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import math
from pathlib import Path

import pickle
import torch
from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from models import Pathology

from utils import get_logger
from proc_utils import getConfiguration

logger = get_logger(__name__)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(961, 50),
            nn.ReLU(),
            torch.nn.Linear(50, 2)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        return self.layers(input)


def getInputAndLabelSets(latentData, width, height):
    labelsTrainSet = []
    inputsTrainSet = []
    for d in latentData:
        labelsTrainSet.append(d[0])
        inputsTrainSet.append(d[1].view(width * height))
    inputsTrainSet = torch.stack(inputsTrainSet)
    labelsTrainSetValue = []
    for label in labelsTrainSet:
        if label == Pathology.NORMAL:
            labelsTrainSetValue.append((0.0, 1.0))
        else:
            labelsTrainSetValue.append((1.0, 0.0))
    labelsTrainSetValue = torch.tensor(labelsTrainSetValue)

    return inputsTrainSet, labelsTrainSetValue


if __name__ == '__main__':
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
    pathTensorBoardx = os.path.join(
        config["root"], config["tensoardboardx_folder"])
    
    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    with open(pathTrain, 'rb') as fp:
        latentTrainData = pickle.load(fp)

    shape = latentTrainData[0][1].shape
    width = shape[3]
    height = shape[2]

    random.shuffle(latentTrainData)
    inputsTrainSet, labelsTrainSetValue = getInputAndLabelSets(
        latentTrainData, width, height)
    logger.info(['inputsTrainSet', inputsTrainSet.shape])

    writer = SummaryWriter(pathTensorBoardx)
    model = Net().to(device)
    criterion = torch.nn.modules.loss.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])

    # TRAIN
    model.train()
    for epoch in range(config['epoch_num']):
        for i in range(0, len(inputsTrainSet), config["batch_size"]):
            optimizer.zero_grad()
            original = inputsTrainSet[i:i + config["batch_size"]]
            labels = labelsTrainSetValue[i:i + config["batch_size"]]
            original.to(device)
            outputs = model(original)
            loss = criterion(outputs, labels)
            writer.add_scalar('Loss/train', loss.data, epoch)
            loss.backward()
            optimizer.step()
        print('train [{}/{}], loss:{:.4f}'.format(
            epoch + 1, config['epoch_num'], loss.data))
    logger.info('Train loss: {}'.format(loss.data))

    # VALIDATE
    with open(pathValidate, 'rb') as fp:
        latentValidateData = pickle.load(fp)
    inputsValidateSet, labelsValidateSetValue = getInputAndLabelSets(
        latentValidateData, width, height)
    logger.info(['inputsValidateSet', inputsValidateSet.shape])

    model.eval()
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for i in range(len(inputsValidateSet)):
        outputs = model(inputsValidateSet[i])
        loss = criterion(outputs, labelsValidateSetValue[i])
        writer.add_scalar('Loss/validate', loss.data, i)

        modelResult = list(outputs).index(max(outputs))
        if int(labelsValidateSetValue[i][0]) == 0:
            # normal
            if modelResult == 0:
                tn += 1
            else:
                fp += 1
        else:
            # hypertrophy
            if modelResult == 0:
                fn += 1
            else:
                tp += 1

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1 = (2*tp) / (2*tp + fp + fn)
    recall = tp / (tp + fn)
    statParametersMsg = "tp: {} fp: {} fn: {} tn: {}".format(tp, fp, fn, tn)
    statMsg = "acc: {} f1: {} recall: {}".format(accuracy, f1, recall)
    logger.info(statParametersMsg)
    logger.info(statMsg)
    print(statParametersMsg)
    print(statMsg)

    Path(pathModel).mkdir(parents=True, exist_ok=True)
    modelPath = os.path.join(
        pathModel, 'classification_fc_la_{}.pt').format(timestr)
    torch.save(model.state_dict(), modelPath)
