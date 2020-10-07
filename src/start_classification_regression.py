# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pickle
import torch
import torch.optim as optim
from torch.autograd import Variable

from utils import get_logger
from proc_utils import getConfiguration

logger = get_logger(__name__)

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

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

    path = os.path.join(config['latent_pickle'], config['latent_file'])
    with open(path, 'rb') as fp:
        latentData = pickle.load(fp)

    shape = latentData[0][1].shape
    width = shape[3]
    height = shape[2]

    inputsTrainSet, labelsTrainSetValue = getInputAndLabelSets(latentData, width, height)

    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    model = LinearRegression()
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    for epoch in range(100):
        model.train()

        optimizer.zero_grad()  # Forward pass
        y_pred = model(input)  # Compute Loss
        loss = criterion(y_pred, labelsTrainSetValue)  # Backward pass
        print('loss: {}'.format(loss.data))

        loss.backward()
        optimizer.step()

