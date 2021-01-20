import numpy as np
from collections import defaultdict


class ColorFactory:

    def __init__(self):
        self.patterns = {}

        self.patterns = defaultdict(self._generate_input)
        self.patterns['sin'] = self._generate_sin_input
        self.patterns['square'] = self._generate_square_input
        self.patterns['circle'] = self._generate_circular_input


    def _generate_circular_input(self,size_x, size_y, scale):
        x = np.arange(0, size_x, 1)
        y = np.arange(0, size_y, 1)
        colors = np.zeros((size_x, size_y, 2))
        for i in x:
            for j in y:
                colors[i][j] = np.array([
                    float(i) / size_y - scale,
                    np.sqrt(((i * scale - (size_x * scale / 2)) ** 2) +
                            ((j * scale - (size_y * scale / 2)) ** 2))])

        return colors.reshape(size_x * size_y,2)


    def _generate_square_input(self,size_x, size_y, offset):
        x = np.arange(0, size_x, 1)
        y = np.arange(0, size_y, 1)
        colors = np.zeros((size_x, size_y, 2))
        for i in x:
            for j in y:
                colors[i][j] = np.array([abs(max([i, j]) / size_x - offset),
                                         abs(max([i, j]) / size_y - offset)])
        return colors.reshape(size_x * size_y, 2)


    def _generate_sin_input(self, size_x, size_y, width):
        x = np.arange(0, size_x, 1)
        y = np.arange(0, size_y, 1)
        colors = np.zeros((size_x, size_y, 2))
        for i in x:
            for j in y:
                colors[i][j] = np.array([np.sin(i * width) * size_x, j])
        return colors.reshape(size_x * size_y, 2)


    def _generate_input(self, size_x, size_y):
        x = np.arange(0, size_x, 1)
        y = np.arange(0, size_y, 1)
        colors = np.zeros((size_x, size_y, 2))
        for i in x:
            for j in y:
                colors[i][j] = np.array([float(i) / size_y - 1.5, float(j) / size_x + 1.5])
        return colors.reshape(self.size_x * self.size_y, 2)


    def generate(self, pattern, **kwargs):
        return self.patterns[pattern](**kwargs)

    
if __name__=="__main__":
    
    color_factory = ColorFactory()
    square_input = color_factory.generate(pattern="square",size_x=256,size_y=256, offset=0)
    circle_input = color_factory.generate(pattern="circle",size_x=256,size_y=256, scale=0.5)
