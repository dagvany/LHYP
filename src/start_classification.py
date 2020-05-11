#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pickle
from torch import nn
import torch.optim as optim
from torch.autograd import Variable

from utils import get_logger
from proc_utils import getConfiguration

logger = get_logger(__name__)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(961, 128),
            nn.Linear(128, 11),
            nn.Softmax()
        )

    def forward(self, input):
        return self.layers(input)

if __name__ == '__main__':
    config = getConfiguration("config_classification.json")

    path = os.path.join(config['latent_pickle'], config['latent_file'])
    with open(path, 'rb') as fp:
        latentData = pickle.load(fp)

    shape = latentData[0][1].shape
    width = shape[3]
    height = shape[2]
    trainSet = ([], [])
    for d in latentData:
        trainSet[1].append(d[0])
        trainSet[0].append(d[1].view(-1, width*height))

    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    for epoch in range(20):
        model.train()
        inputs = Variable(trainSet[0])
        labels = Variable(trainSet[1])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        print('loss: {}'.format(loss.data))

        loss.backward()
        optimizer.step()
