import os
import math
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def draw_squares(squares):
    fig, axes = plt.subplots(6, 20, figsize=(18, 7),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        X, y = squares[i]
        ax.imshow(-1 * (X.reshape(3, 3) - 255), cmap='gray')
        ax.set_title('{:.0f} {:.0f} {:.0f}'.format(y[0], y[1], y[2]))