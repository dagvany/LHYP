#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import get_logger

logger = get_logger(__name__)

class AutoEncoder(nn.Module):
    def __init__(self, inputLayerSize, intermediateLayerSize, latentLayerSize):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inputLayerSize, intermediateLayerSize),
            nn.ReLU(True),
            nn.Linear(intermediateLayerSize, intermediateLayerSize),
            nn.ReLU(True),
            nn.Linear(intermediateLayerSize, latentLayerSize))

        self.decoder = nn.Sequential(
            nn.Linear(latentLayerSize, intermediateLayerSize),
            nn.ReLU(True),
            nn.Linear(intermediateLayerSize, intermediateLayerSize),
            nn.ReLU(True),
            nn.Linear(intermediateLayerSize, inputLayerSize))

    def forward(self, bachedInputs):
        encoded = self.encoder(bachedInputs)
        #for l in encoded:
        #    logger.info(l[0])
        decoded = self.decoder(encoded)
        return decoded

def run(
    images,
    height,
    width,
    intermediateLayerSize,
    latentLayerSize,
    numEpochs,
    batchSize,
    learningRate):
    inputLayerSize = height * width
    model = AutoEncoder(height*width, inputLayerSize, latentLayerSize).cpu()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learningRate,
        weight_decay=1e-5)
    
    images = torch.Tensor(images)
    path = '/media/Data/Dipterv_MRI/test/ae/{:04d}_{}_{}.png'
    pathFinal = '/media/Data/Dipterv_MRI/test/ae/final_{}_{}.png'
    #for epoch in range(numEpochs):
    original = images[len(images)-batchSize:len(images)].view(-1, 1, height*width)
    _saveMaxrixToImg(original, height, width,
        pathFinal.format(0, 'orig', 0))
    epoch = -1
    while True:
        epoch = epoch + 1
        for i in range(0, len(images), batchSize):
            original = images[i:i+batchSize].view(-1, 1, height*width)
            # forward
            output = model(original)
            loss = criterion(output, original)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, numEpochs, loss.data))
        if (epoch+1) % 10 == 0:
            _saveMaxrixToImg(output.cpu().data, height, width,
                path.format(epoch+1, 'res', loss.data))
        if loss.data < 0.05:
            break
    _saveMaxrixToImg(output.cpu().data, height, width,
        pathFinal.format(epoch+1, 'res', loss.data))

def _convertToImg(vector, height, width):
    vector = 0.5 * (vector + 1)
    vector = vector.clamp(0, 1)
    img = vector.view(vector.size(0), 1, height, width)
    return img

def _saveMaxrixToImg(matrix, height, width, path):
    pic = _convertToImg(matrix, height, width)
    save_image(pic, path)