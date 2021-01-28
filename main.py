from CPPN.network import NN, plot_image, plot_input
from color_factory import ColorFactory
from scipy import stats
import torch
from pprint import pprint

# Constants
SIZE_X = 1080
SIZE_Y = 1080
LAYERS = 16
NEURONS = 32
INPUT_SIZE = 2
SCALE = 0.1


if __name__ == '__main__':

    factory = ColorFactory()
    net_3 = NN(input_size=INPUT_SIZE, n_neurons=NEURONS, n_layers=LAYERS, output_size=3)
    net_4 = NN(input_size=INPUT_SIZE, n_neurons=NEURONS, n_layers=LAYERS, output_size=4)
    net_small_3 = NN(input_size=INPUT_SIZE, n_neurons=16, n_layers=8, output_size=3)
    net_small_4 = NN(input_size=INPUT_SIZE, n_neurons=16, n_layers=8, output_size=4)
    net_big_3 = NN(input_size=INPUT_SIZE, n_neurons=64, n_layers=32, output_size=3)

    # Simple input
    simple_input = factory.generate(pattern='simple', size_x=SIZE_X, size_y=SIZE_Y, scale=0)
    pprint((simple_input[:, 0] * 255 / SIZE_X).reshape((SIZE_X, SIZE_Y)))
    pprint((simple_input[:, 1] * 255 / SIZE_X).reshape((SIZE_X, SIZE_Y)))
    stats.describe(simple_input)
    plot_input(simple_input, SIZE_X, SIZE_Y, path='images/simple_input')

    # Output 3
    output = net_small_3(torch.tensor(simple_input).type(torch.FloatTensor))
    print(output)
    plot_image(output, SIZE_X, SIZE_Y, 3,
               f'images/simple_output-l{8}-n{16}-size{SIZE_X}-o3')

    # Output 4
    output = net_small_4(torch.tensor(simple_input).type(torch.FloatTensor))
    print(output)
    plot_image(output, SIZE_X, SIZE_Y, 4,
               f'images/simple_output-l{8}-n{16}-size{SIZE_X}-o4')


    # Circular input
    circular_input = factory.generate(pattern='circle', size_x=SIZE_X, size_y=SIZE_Y, scale=0.01)
    stats.describe(circular_input)
    plot_input(circular_input, SIZE_X, SIZE_Y, path='images/circle_input')

    # Output 3
    output = net_small_3(torch.tensor(circular_input).type(torch.FloatTensor))
    plot_image(output, SIZE_X, SIZE_Y, 3,
               f'images/circular_input-l{8}-n{16}-size{SIZE_X}-o3-s001')

    # Output 4
    output = net_4(torch.tensor(circular_input).type(torch.FloatTensor))
    plot_image(output, SIZE_X, SIZE_Y, 4,
               f'images/circular_input-l{LAYERS}-n{NEURONS}-size{SIZE_X}-o4-s001')

    # Better resolution
    circular_input = factory.generate(pattern='circle', size_x=1920, size_y=1920, scale=0.01)
    stats.describe(circular_input)
    plot_input(circular_input, 1920, 1920, path='images/circle_input')

    # Output 3
    output = net_3(torch.tensor(circular_input).type(torch.FloatTensor))
    plot_image(output, 1920, 1920, 3,
               f'images/circular_input-l{LAYERS}-n{NEURONS}-size{1920}-o3-s001')

    # Less layers
    output = net_small_3(torch.tensor(circular_input).type(torch.FloatTensor))
    plot_image(output, 1920, 1920, 3,
               f'images/circular_input-l{8}-n{16}-size{1920}-o3-s001')

    # Other scale
    circular_input = factory.generate(pattern='circle', size_x=SIZE_X, size_y=SIZE_Y, scale=0.0001)
    stats.describe(circular_input)
    plot_input(circular_input, SIZE_X, SIZE_Y, path='images/circle_input-00001')

    # Output 3
    output = net_3(torch.tensor(circular_input).type(torch.FloatTensor))
    plot_image(output, SIZE_X, SIZE_Y, 3,
               f'images/circular_input-l{LAYERS}-n{NEURONS}-size{SIZE_X}-o3-s00001')


    # Square
    square_input = factory.generate(pattern='square', size_x=SIZE_X, size_y=SIZE_Y, scale=0.5)
    stats.describe(square_input)
    plot_input(square_input, SIZE_X, SIZE_Y, path='images/square_input-05')

    # Output 3
    output = net_small_3(torch.tensor(square_input).type(torch.FloatTensor))
    plot_image(output, SIZE_X, SIZE_Y, 3,
               f'images/square_input-l{LAYERS}-n{NEURONS}-size{SIZE_X}-o4-05')

    # Parania
    paranoia_input = factory.generate(pattern='paranoia', size_x=SIZE_X, size_y=SIZE_Y, scale=0.01)
    plot_input(paranoia_input, SIZE_X, SIZE_Y, path='images/paranoia_input-001')

    # Output 3
    output = net_3(torch.tensor(paranoia_input).type(torch.FloatTensor))
    plot_image(output, SIZE_X, SIZE_Y, 3,
               f'images/paranoia_input-l{LAYERS}-n{NEURONS}-size{SIZE_X}-o3-001')