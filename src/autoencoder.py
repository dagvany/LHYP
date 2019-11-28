#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import time
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
    learningRate,
    imgFolder,
    modelFolder,
    cudaSeed):
    inputLayerSize = height * width
    if cudaSeed > 0:
        device = 'cuda'
        torch.manual_seed(cudaSeed)
    else:
        device = 'cpu'

    model = AutoEncoder(height*width, inputLayerSize, latentLayerSize).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learningRate,
        weight_decay=1e-5)
    
    images = torch.Tensor(images)
    path = os.path.join(imgFolder, 'ae/{:04d}_{}_{}.png')
    pathFinal = os.path.join(imgFolder, 'final_{}_{}.png')
    original = images[len(images)-batchSize:len(images)].view(-1, 1, height*width)
    _saveMaxrixToImg(original, height, width,
        pathFinal.format(0, 'orig', 0))
    for epoch in range(numEpochs):
        for i in range(0, len(images), batchSize):
            original = images[i:i+batchSize].view(-1, 1, height*width)
            data = original.to(device)
            # forward
            output = model(data)
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
    _saveMaxrixToImg(output.cpu().data, height, width,
        pathFinal.format(epoch+1, 'res', loss.data))
    
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    modelPath = os.path.join(modelFolder, 'ae_{}.pt').format(timestr)
    torch.save(model.state_dict(), modelPath)

def _convertToImg(vector, height, width):
    vector = 0.5 * (vector + 1)
    vector = vector.clamp(0, 1)
    img = vector.view(vector.size(0), 1, height, width)
    return img

def _saveMaxrixToImg(matrix, height, width, path):
    pic = _convertToImg(matrix, height, width)
    save_image(pic, path)