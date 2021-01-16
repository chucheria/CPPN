from network import NN, plot_callback, plot_image
from generator import Generator


OUTPUT_SIZE = 3
SIZE_X, SIZE_Y = 256, 256
SCALE = 0.1
NEURONS = 8
LAYERS = 16

net = NN(n_neurons=NEURONS, n_layers=LAYERS, output_size=OUTPUT_SIZE)
generator = Generator(size_x=SIZE_X, size_y=SIZE_Y, scale=SCALE)
gen = generator.patterns['square']()

plot_image(plot_callback(net, gen, size_x=SIZE_X, size_y=SIZE_Y, output_size=OUTPUT_SIZE))

for n in range(0, 11):
    net.train(gen, n_steps=n)
    plot_image(plot_callback(net, gen, size_x=SIZE_X, size_y=SIZE_Y, output_size=OUTPUT_SIZE))