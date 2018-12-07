#%%
import os
import math
import shutil
import numpy as np
from PIL import Image
from utils.draw import draw_xy
import matplotlib.pyplot as plt

# generate integers!
def generate(count):
    X = np.random.randint(0, high=255, size=(count, 9))
    a = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 1, 1, 1, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 1, 1, 1]])

    Y = np.eye(3)[np.argmax(X.dot(a.T), axis=1)]
    return X, Y

#%% are we making the right things?
draw_xy(*generate(120))

#%% run training loop
def train(n=512, epochs=56000, lr = 1e-3):
    # weight vector
    W = np.random.randn(9, 3)
    b = np.random.randn(1, 3)

    # training loop
    for t in range(epochs):
        # get new training data
        X, y = generate(n)
        X = X / 255

        # model function
        h = X.dot(W) + b

        # compute loss
        loss = np.square(h - y).mean()

        # compute accuracy
        acc = (np.argmax(h, axis=1) == np.argmax(y, axis=1)).mean()

        if t % 5000 == 0:
            print('l: {:>8f}, a {:>.4f} (e {})'.format(loss, acc, t))

        # grad + update
        grad_w = 2 * X.T.dot(h - y) / n
        W -= lr * grad_w

        grad_b = 2 * np.sum(h - y, axis=0) / n
        b -= lr * grad_b

    return W, b

#%%
W, b = train()
print('\nFinal W = \n\n{}\n\nFinal b = \n\n{}'.format(W, b))

#%%
npix = np.array([[255, 143, 23, 255, 187, 93, 255, 255, 255]])
a = npix.dot(W) + b
d = ['top', 'middle', 'bottom']
print(d[np.argmax(a)])
