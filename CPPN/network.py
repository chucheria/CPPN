import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from color_factory import ColorFactory


def save(colors, name):
    plt.imsave(f"images/{name}.png", colors)


def plot_image(colors, size_x, size_y):
    img = colors.detach().numpy()
    img = img.reshape(size_x, size_y, 3) * 255
    plt.figure(figsize=(6, 6))
    plt.imshow(img.astype(np.uint8))


class NN(nn.Module):

    @staticmethod
    def init(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight)

    def __init__(self, activation=nn.Tanh, input_size=2, n_neurons=2, n_layers=9, output_size=3):
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


if __name__ == '__main__':

    from functools import partial

    # Generate patterns
    square_input = ColorFactory().generate(pattern="square", size_x=256, size_y=256, scale=0)
    circle_input = ColorFactory().generate(pattern="circle", size_x=256, size_y=256, scale=0.5)

    net = NN(input_size=2, n_neurons=20, n_layers=20, output_size=3)

    square_img = net(torch.tensor(square_input).type(torch.FloatTensor))
    circle_img = net(torch.tensor(circle_input).type(torch.FloatTensor))

    #plot_image(square_img, 256, 256)
    plot_image(circle_img, 256, 256)

    # Create target pattern
    target_input = square_img
    target_input[:, 0] = 0.
    target_input[:, 1] = 0.

    plot_image(target_input, 256, 256)

    # Train training pattern
    callback_plot = partial(plot_image, size_x=256, size_y=256)

    source = square_input
    target = target_input.detach().numpy()

    net.train(source, target, 20, callback_plot)
