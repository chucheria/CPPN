import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import defaultdict


def plot_image(colors, fig_size=6):
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(colors, interpolation='nearest', vmin=0, vmax=1)


def plot_callback(net, input, size_x, size_y, output_size=3):
    img = net(torch.tensor(input).type(torch.FloatTensor)).detach().numpy()
    return img.reshape(size_x, size_y, output_size)


class NN(nn.Module):

    @staticmethod
    def _init(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight)
    
    def __init__(self, activation=nn.Tanh, input_size=2, n_neurons=2, n_layers=9, output_size=3):
        super(NN, self).__init__()

        layers = []
        layers += [nn.Linear(input_size, n_neurons, bias=True)]
        layers += [activation()]

        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_layers, bias=False)]
            layers += [activation()]

        layers += [nn.Linear(n_neurons, output_size, bias=False), nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)
        self.apply(self._init)
        
    @property
    def optimizer(self):
        _opt = torch.optim.SGD(self.layers.parameters(), lr=0.1, momentum=0.9)
        _opt.zero_grad()
        return _opt
    
    @property
    def loss(self):
        return torch.mean
    
    def forward(self, x):
        return self.layers(x)

    def train(self, x, n_steps=10, callback=lambda x: None):

        self.apply(self._init)
        opt = self.optimizer

        for _ in range(n_steps):

            img = self.layers(torch.tensor(x).type(torch.FloatTensor))
            loss = self.loss(img)
    
            callback(img)

            loss.backward()
            opt.step()
