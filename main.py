from network import NN, plot_callback, plot_image
from generator import Generator
import torch.nn as nn


OUTPUT_SIZE = 4
SIZE_X, SIZE_Y = 1080, 1080
SCALE = 0.1
NEURONS = 16
LAYERS = 9
ACTIVATION = nn.Tanh

net = NN(n_neurons=NEURONS, n_layers=LAYERS, output_size=OUTPUT_SIZE, activation=ACTIVATION, input_size=4)
generator = Generator(size_x=SIZE_X, size_y=SIZE_Y, scale=SCALE)
gen = generator.patterns['latent']()
#net.train(gen)

plot_image(plot_callback(net, gen, size_x=SIZE_X, size_y=SIZE_Y, output_size=OUTPUT_SIZE))
#
# for n in range(0, 11):
#     print(n)
#     print('---------')
#     net.train(gen, n_steps=n)
#     plot_image(plot_callback(net, gen, size_x=SIZE_X, size_y=SIZE_Y, output_size=OUTPUT_SIZE))
#