from CPPN.network import NN, plot_image, plot_input
from color_factory import ColorFactory
from scipy import stats
import torch

# Constants
SIZE_X = 256
SIZE_Y = 256
LAYERS = 12
NEURONS = 32
OUTPUT_SIZE = 3
INPUT_SIZE = 2
SCALE = 0.1


def process(pattern):

    input_colors = factory.generate(pattern=pattern, size_x=SIZE_X, size_y=SIZE_Y, scale=SCALE)
    print(input_colors)
    print(stats.describe(input_colors))
    plot_input(input_colors, SIZE_X, SIZE_Y)

    output = net(torch.tensor(input_colors).type(torch.FloatTensor))
    print(output)
    plot_image(output, SIZE_X, SIZE_Y)


if __name__ == '__main__':

    factory = ColorFactory()
    net = NN(input_size=INPUT_SIZE, n_neurons=NEURONS, n_layers=LAYERS, output_size=OUTPUT_SIZE)

    # Simple input
    process('simple')
    # Scaled
    process('scaled')
    # Circle
    process('circle')
    # Square
    process('square')
    # Sin
    process('sin')
    # Paranoia
    process('paranoia')