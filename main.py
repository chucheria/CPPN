from network import NN, plot_image
from functools import partial
from color_factory import ColorFactory
import torch.nn as nn
import torch


OUTPUT_SIZE = 4
SIZE_X, SIZE_Y = 1080, 1080
SCALE = 0.1
NEURONS = 16
LAYERS = 9
ACTIVATION = nn.Tanh

# Generate patterns
square_input = ColorFactory().generate(pattern="square", size_x=SIZE_X, size_y=SIZE_X, scale=0)
circle_input = ColorFactory().generate(pattern="circle", size_x=256, size_y=256, scale=0.5)

net = NN(input_size=2, n_neurons=20, n_layers=20, output_size=3)

square_img = net(torch.tensor(square_input).type(torch.FloatTensor))
circle_img = net(torch.tensor(circle_input).type(torch.FloatTensor))

plot_image(square_img, 256, 256)
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