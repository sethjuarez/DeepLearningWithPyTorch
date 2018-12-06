#%%
import os
import math
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.draw import draw_single

# generate integers!
def generate(count):
    X = np.random.randint(0, high=255, size=(count, 9))
    Y = X.dot(np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]))
    Y[Y > 0] = 1
    Y[Y < 0] = -1
    return X, Y

#%% are we making the right things?
draw_single(*generate(120))

#%% run training loop
def train(n=512, epochs=31000, lr = 1e-3):
    # weight vector
    W = np.random.randn(9, 1)

    # training loop
    for t in range(epochs):
        # get new training data
        X, y = generate(n)
        X = X / 255
        y = y.reshape(n, 1)

        # model function
        h = X.dot(W)

        # compute loss
        loss = np.square(h - y).mean()

        # compute accuracy
        acc = (np.sign(h) == y).mean()

        if t % 5000 == 0:
            print('l: {:>8f}, a {:>.4f} (e {})'.format(loss, acc, t))

        # grad + update
        grad_w = 2 * X.T.dot(h - y) / n
        W -= lr * grad_w

    return W

#%%
W = train()
print('\nFinal W = \n\n{}'.format(W))