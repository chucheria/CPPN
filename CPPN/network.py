import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class NN(nn.Module):

    @staticmethod
    def init(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight)

    def __init__(self, activation=nn.Tanh, input_size=2, n_neurons=2, n_layers=9, output_size=3):
        torch.manual_seed(4308271371631272292)
        super(NN, self).__init__()

        layers = []
        layers += [nn.Linear(input_size, n_neurons, bias=True)]
        layers += [activation()]

        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons, bias=True)]
            layers += [activation()]

        layers += [nn.Linear(n_neurons, output_size, bias=True), nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)
        self.apply(self.init)

    @property
    def optimizer(self):
        _opt = torch.optim.SGD(self.layers.parameters(), lr=0.1, momentum=0.9)
        _opt.zero_grad()
        return _opt

    @property
    def criterion(self):
        return nn.MSELoss()

    @property
    def loss(self):
        def _loss(result, target):
            error = self.criterion(result, target)
            print("loss: ", error.data)

            return error * 0.01

        return _loss

    def forward(self, x):
        return self.layers(x)

    def train(self, source, target, n_steps=10, callback=None):

        opt = self.optimizer
        target = torch.tensor(target)

        for _ in range(n_steps):
            result = self.forward(torch.tensor(source).type(torch.FloatTensor))

            _ = callback(result) if callable(callback) else None

            loss = self.loss(result, target)
            loss.backward()
            opt.step()
            
        return result
