import os
import math
import torch
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

def draw_digits(digits):
    fig, axes = plt.subplots(6, 20, figsize=(18, 7),
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        X, y = digits[i]
        ax.imshow(255 - X.reshape(28,28) * 255, cmap='gray')
        ax.set_title('{:.0f}'.format(torch.argmax(y).item()))

def draw_xy(X, y):
    fig, axes = plt.subplots(6, 20, figsize=(18, 7),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(-1 * (X[i].reshape(3, 3) - 255), cmap='gray')
        ax.set_title('{:.0f} {:.0f} {:.0f}'.format(y[i][0], y[i][1], y[i][2]))

def draw_single(X, y):
    fig, axes = plt.subplots(6, 20, figsize=(18, 7),
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(-1 * (X[i].reshape(3, 3) - 255), cmap='gray')
        ax.set_title('{:.0f}'.format(y[i]))
