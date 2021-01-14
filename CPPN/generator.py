import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os, copy
from PIL import Image
from IPython import display

class NN(nn.Module):
    
    def _init_(self, activation=nn.Tanh, input_size=2, n_neurons=2, n_layers=9, output_size=3):

        super(NN, self)._init_()
        layers = []

        layers += [nn.Linear(input_size, num_neurons, bias=True)]
        layers += [activation()]

        for _ in range(num_layers - 1):
            layers += [nn.Linear(num_neurons, num_neurons, bias=False)]
            layers += [activation()]

        layers += [nn.Linear(num_neurons, output_size, bias=False), nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)
        self.optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        self.loss = torch.mean 
        self.apply(self._init)
            
    @staticmethod
    def _init(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight)

    def forward(self, x):
        return self.layers(x)
 
    def train(self, x, n_steps=10, callback=lambda x: None):
 
        self.optimizer.zero_grad()
        for _ in range(n_steps):

            img = net(torch.tensor(x).type(torch.FloatTensor))
            loss = self.loss(img)
    
            callback(img)

            loss.backward()
            optimizer.step()


class Generator():

    from collections import defaultdict

    def _init_(self):
        self.patterns = defaultdict(self.generate_input)

        self.patterns['sin'] = generate_sin_input
        self.patterns['square'] = generate_square_input
        self.patterns['circular'] = self.generate_circular_input

    def _generate_circular_input(size_x, size_y, scale):
        pass

    def _generate_circular_input(size_x, size_y, offset):
        pass

    def _generate_sin_input(size_x, size_y, width):
        pass

    def _generate_input(size_x, size_y):
        pass

    def generate(pattern, **kwargs):
        return self.patters[pattern](**kwargs)


def plot_callback(net, input, size_x, size_y, output_size=3):
    img = net(torch.tensor(input).type(torch.FloatTensor)).detach().numpy()
    img.reshape(size_x, size_y, output_size)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(colors, interpolation='nearest', vmin=0, vmax=1)
