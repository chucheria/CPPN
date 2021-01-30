import numpy as np
from collections import defaultdict


class ColorFactory:

    def __init__(self):
        self.patterns = defaultdict(dict)
        self.patterns['simple'] = self._generate_simple_input
        self.patterns['sin'] = self._generate_sin_input
        self.patterns['square'] = self._generate_square_input
        self.patterns['circle'] = self._generate_circular_input
        self.patterns['scaled'] = self._generate_scaled_input
        self.patterns['paranoia'] = self._generate_paranoia_input

    @staticmethod
    def _generate_circular_input(i, j, size_x, size_y, scale):
        return np.array([float(i) / size_y - scale,
                         np.sqrt(((i * scale - (size_x * scale / 2)) ** 2) +
                                 ((j * scale - (size_y * scale / 2)) ** 2))])

    @staticmethod
    def _generate_square_input(i, j, size_x, size_y, scale):
        return np.array([abs(max([i, j]) / size_x - scale),
                         abs(max([i, j]) / size_y - scale)])

    @staticmethod
    def _generate_sin_input(i, j, size_x, size_y, scale):
        return np.array([np.sin(i * scale) * size_x, j])

    @staticmethod
    def _generate_scaled_input(i, j, size_x, size_y, scale):
        return np.array([float(i) / size_y - scale, float(j) / size_x + scale])

    @staticmethod
    def _generate_simple_input(i, j, size_x, size_y, scale):
        return np.array([i, j])

    @staticmethod
    def _generate_paranoia_input(i, j, size_x, size_y, scale):
        return np.array([np.sin(i * scale) * 2 * np.pi/size_x, np.sin(j * scale) * 2 * np.pi/size_y])

    def generate(self, pattern, size_x, size_y, **kwargs):

        x = np.arange(0, size_x, 1)
        y = np.arange(0, size_y, 1)

        n_features = 2
        n_colors = size_x * size_y

        colors = np.zeros((n_colors, n_features))

        for i in x:
            for j in y:
                colors[i * size_y + j] = self.patterns[pattern](i, j, size_x, size_y, **kwargs)

        return colors
