#%%
import os
import math
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# generate integers!
def generate(count):
    X = np.random.randint(0, high=255, size=(count, 9))
    Y = X.dot(np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]))
    Y[Y > 0] = 1
    Y[Y < 0] = 0
    return X, Y

#%% 
# are we making the right things?
X, y = generate(120)
fig, axes = plt.subplots(6, 20, figsize=(18, 7),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
   ax.imshow(-1 * (X[i].reshape(3, 3) - 255), cmap='gray')
   ax.set_title('{:.0f}'.format(y[i]))

#%% run training loop
def train(n=512, epochs=50000, lr = 1e-3):
    # weight vector
    W = np.random.randn(9, 1)

    # training loop
    for t in range(epochs):
        # get new training data
        X, y = generate(n)
        X = X / 255
        y = y.reshape(n, 1) * 2 - 1

        # model function
        h = X.dot(W)

        # compute loss
        loss = np.square(h - y).mean()

        # compute accuracy
        acc = (np.sign(h) == y).mean()

        if t % 5000 == 0:
            print('loss: {:>8f}, accuracy {:>.4f} (epoch {})'.format(loss, acc, t))

        # no more to do
        if acc >= 1:
            print('\nStopping:\nloss: {:>8f}, accuracy {:>.4f} (epoch {})'.format(loss, acc, t))
            break

        # grad + update
        grad_w = 2 * X.T.dot(h - y) / n
        W -= lr * grad_w

    print('\nFinal W = \n\n{}'.format(W))
    return W

#%%
model = train()