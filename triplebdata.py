#%%
import os
import torch
import math
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from squaredata import SquareDataset
from torch.utils.data import DataLoader

#%% run training loop
def train(n=512, epochs=50000, lr = 1e-3):
    # weight vector
    W = torch.randn(9, 3, dtype=torch.float)
    b = torch.randn(1, 3, dtype=torch.float)

    dataset = SquareDataset(60000)
    dataloader = DataLoader(dataset, batch_size=n)

    # training loop
    for batch, sample in enumerate(dataloader):
        # get new training data
        X, y = sample
        X = X / 255

        # model function
        h = X.mm(W) + b

        # compute loss
        loss = (h - y).pow(2).sum().item()

        # compute accuracy
        acc = (h.argmax(1) == y.argmax(1)).type(torch.float).mean().item()

        if batch % 5000 == 0:
            print('loss: {:>8f}, accuracy {:>.4f} (epoch {})'.format(loss, acc, batch))

        # no more to do
        if acc >= 1:
            print('\nStopping:\nloss: {:>8f}, accuracy {:>.4f} (epoch {})'.format(loss, acc, batch))
            break

        # grad + update
        grad_w = 2 * X.transpose(0, 1).mm(h - y) / n
        W -= lr * grad_w

        grad_b = 2 * torch.sum(h - y, axis=0) / n
        b -= lr * grad_b

    print('\nFinal W = \n\n{}'.format(W))
    print('\nFinal b = \n\n{}'.format(b))
    return W, b

#%%
model = train()
