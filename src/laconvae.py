#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn


class LaConvAE(nn.Module):
    def __init__(self):
        super(LaConvAE, self).__init__()

        self.encoderConv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=0),
            nn.Softmax2d(),

            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=0),
            nn.Softmax2d(),

            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.Softmax2d(),

            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=0),
            nn.Softmax2d(),

            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=0),
            nn.Softmax(),

            # Instead of MaxPool
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
            nn.Softmax2d(),
        )

        self.decoderDeConv = nn.Sequential(
            nn.ConvTranspose2d(1, 8, kernel_size=1, stride=1, padding=0),
            nn.Softmax2d(),

            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=0),
            nn.Softmax2d(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.Softmax2d(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=1, padding=0),
            nn.Softmax2d(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=1, padding=0),
            nn.Softmax2d(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=1, padding=0),
            nn.Softmax2d(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    #nn.init.uniform(m.bias)
                    nn.init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    #nn.init.uniform(m.bias)
                    nn.init.xavier_uniform(m.weight)


    def forward(self, bachedInputs):
        latent = self.encoderConv(bachedInputs)
        decoded = self.decoderDeConv(latent)
        return decoded, latent
