import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


FIG_SIZE_X = 8
FIG_SIZE_Y = 8


def plot_image(colors, size_x, size_y, output_size=3, path=None):
    img = colors.detach().numpy()
    img = img.reshape(size_x, size_y, output_size) * 255

    plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
    plt.axis('off')
    plt.imshow(img.astype(np.uint8))
    plt.show()

    if path:
        plt.imsave(f'{path}.png', img.astype(np.uint8), dpi=300)


def plot_input(colors, size_x, size_y, path=None):

    def steps(ft, c, x, y, p):
        feature = c[:, ft]
        feature = feature - np.min(feature)
        feature = feature * 255 / np.max(feature)
        feature = feature.astype(np.uint8).reshape((x, y))
        plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
        plt.axis('off')
        plt.imshow(feature, cmap='Greys')
        if p:
            plt.imsave(f'{p}_{ft}.png', feature, cmap='Greys', dpi=300)

    steps(0, colors, size_x, size_y, path)
    steps(1, colors, size_x, size_y, path)
    
    
def plot_pattern(pattern, size_x, size_y, path=None):
    
    for i in range(2):
        
        plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
        plt.axis('off')
        
        img = pattern[:,i]
        img = 255*img.reshape((size_x,size_y))

        plt.imshow(img, cmap='Greys')
        plt.show()
        
        if path:
            plt.imsave(f'{path}_{i}.png', img.astype(np.uint8), dpi=300)
