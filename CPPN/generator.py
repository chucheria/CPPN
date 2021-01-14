import numpy as np


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
