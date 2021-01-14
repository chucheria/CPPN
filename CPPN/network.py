import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import defaultdict


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


class Generator():

    def __init__(self, pattern, **kwargs):
        self.__dict__.update((key, value) for key, value in kwargs.items())
        self.patterns = defaultdict(self._generate_input(pattern))
        self.patterns['sin'] = self._generate_sin_input()
        self.patterns['square'] = self._generate_square_input()
        self.patterns['circular'] = self._generate_circular_input()

    def _generate_circular_input(self):
        x = np.arange(0, self.size_x, 1)
        y = np.arange(0, self.size_y, 1)
        colors = np.zeros((self.size_x, self.size_y, 3))
        for i in x:
            for j in y:
                colors[i][j] = np.array([
                    float(i) / self.size_y - self.scale,
                    float(j) / self.size_y - self.scale,
                    np.sqrt(((i * self.scale - (self.size_x * self.scale / 2)) ** 2) +
                            ((j * self.scale - (self.size_y * self.scale / 2)) ** 2))])

        return colors.reshape(self.size_x * self.size_y, 3)

    def _generate_square_input(self):
        x = np.arange(0, self.size_x, 1)
        y = np.arange(0, self.size_y, 1)
        colors = np.zeros((self.size_x, self.size_y, 2))
        for i in x:
            for j in y:
                colors[i][j] = np.array([abs(max([i, j]) / self.size_x - self.offset),
                                         abs(max([i, j]) / self.size_y - self.offset)])
        return colors.reshape(self.size_x * self.size_y, 2)

    def _generate_sin_input(self):
        x = np.arange(0, self.size_x, 1)
        y = np.arange(0, self.size_y, 1)
        colors = np.zeros((self.size_x, self.size_y, 2))
        for i in x:
            for j in y:
                colors[i][j] = np.array([np.sin(i * self.width) * self.size_x, j])
        return colors.reshape(self.size_x * self.size_y, 2)

    def _generate_input(self):
        x = np.arange(0, self.size_x, 1)
        y = np.arange(0, self.size_y, 1)
        colors = np.zeros((self.size_x, self.size_y, 2))
        for i in x:
            for j in y:
                colors[i][j] = np.array([float(i) / self.size_y - 1.5, float(j) / self.size_x + 1.5])
                # colors[i][j] = np.array([np.sin(i/(size_x/5)), np.cos(j/(size_y/5))])
                # colors[i][j] = np.array(np.sin(i), np.cos(j))
                # colors[i][j] = np.random.normal(float(i) / size_y, float(j) / size_x, size=(1, 2))
        return colors.reshape(self.size_x * self.size_y, 2)

    def generate(self, pattern):
        return self.patterns[pattern]()


def plot_image(colors, fig_size=6):
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(colors, interpolation='nearest', vmin=0, vmax=1)


def plot_callback(net, input, size_x, size_y, output_size=3):
    img = net(torch.tensor(input).type(torch.FloatTensor)).detach().numpy()
    return img.reshape(size_x, size_y, output_size)
