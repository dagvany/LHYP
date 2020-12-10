#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn


class LaConvAEsmall(nn.Module):
    def getNameOfActivation(self):
        return self.activation()._get_name()

    def __init__(self):
        super(LaConvAEsmall, self).__init__()

        self.activation = nn.ReLU

        self.encoderConv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=0),
            self.activation(),

            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0),
            self.activation(),

            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            self.activation(),

            # Instead of MaxPool
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            self.activation(),
        )

        self.decoderDeConv = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=1, stride=1, padding=0),
            self.activation(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0),
            self.activation(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0),
            self.activation(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=0),
            self.activation()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.xavier_uniform(m.weight)

    def forward(self, bachedInputs):
        latent = self.encoderConv(bachedInputs)
        decoded = self.decoderDeConv(latent)
        return decoded, latent
