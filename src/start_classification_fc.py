#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
from pathlib import Path

import pickle
import torch
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
            torch.nn.Linear(961, 300),
            torch.nn.Linear(300, 10),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.layers(input)

def getInputAndLabelSets(latentData, width, height):
    labelsTrainSet = []
    inputsTrainSet = []
    for d in latentData:
        labelsTrainSet.append(d[0])
        inputsTrainSet.append(d[1].view(width * height))
    inputsTrainSet = torch.stack(inputsTrainSet)
    labelsTrainSetValue = torch.tensor([l.value for l in labelsTrainSet])

    return inputsTrainSet, labelsTrainSetValue

if __name__ == '__main__':
    config = getConfiguration("config_classification.json")

    pathTrain = os.path.join(config['latent_pickle'], config['latent_train_file'])
    with open(pathTrain, 'rb') as fp:
        latentTrainData = pickle.load(fp)

    shape = latentTrainData[0][1].shape
    width = shape[3]
    height = shape[2]

    random.shuffle(latentTrainData)
    inputsTrainSet, labelsTrainSetValue = getInputAndLabelSets(latentTrainData, width, height)
    logger.info(['inputsTrainSet', inputsTrainSet.shape])

    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(config["tensoardboardx_folder"], timestr))
    model = Net().to(device)
    criterion = torch.nn.modules.loss.MultiMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

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
        print('train [{}/{}], loss:{:.4f}'.format(epoch + 1, config['epoch_num'], loss.data))
    logger.info('Train loss: {}'.format(loss.data))

    # VALIDATE
    pathValidate = os.path.join(config['latent_pickle'], config['latent_validate_file'])
    with open(pathValidate, 'rb') as fp:
        latentValidateData = pickle.load(fp)
    inputsValidateSet, labelsValidateSetValue = getInputAndLabelSets(latentValidateData, width, height)
    logger.info(['inputsValidateSet', inputsValidateSet.shape])

    model.eval()
    numOfSuccess = 0
    stat = {}
    for i in range(len(inputsValidateSet)):
        outputs = model(inputsValidateSet[i])
        loss = criterion(outputs, labelsValidateSetValue[i])
        writer.add_scalar('Loss/validate', loss.data, i)

        modelResult = list(outputs).index(max(outputs))
        success = labelsValidateSetValue[i] == modelResult
        if success: numOfSuccess += 1


        expectedValue = int(labelsValidateSetValue[i])
        if Pathology(expectedValue) not in stat:
            stat[Pathology(expectedValue)] = {'success': 0, 'fail': {}}
        if success:
            stat[Pathology(expectedValue)]['success'] += 1
        elif Pathology(modelResult) in stat[Pathology(expectedValue)]['fail']:
            stat[Pathology(expectedValue)]['fail'][Pathology(modelResult)] += 1
        else:
            stat[Pathology(expectedValue)]['fail'].update({Pathology(modelResult): 1})


        print('validate loss: {}'.format(loss.data))
        logger.info('validate  success: {} loss: {}, model: {} label: {}'.format(
            success, loss.data, Pathology(modelResult), Pathology(expectedValue)))
    logger.info("{}/{} ->{}".format(numOfSuccess, len(inputsValidateSet), numOfSuccess/len(inputsValidateSet)))
    print("{}/{} ->{}".format(numOfSuccess, len(inputsValidateSet), numOfSuccess / len(inputsValidateSet)))
    print(stat)
    logger.info(stat)


    Path(config["pytorch_model_folder"]).mkdir(parents=True, exist_ok=True)
    modelPath = os.path.join(config["pytorch_model_folder"], 'classification_fc_la_{}.pt').format(timestr)
    torch.save(model.state_dict(), modelPath)
