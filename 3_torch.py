#%%
import torch
from utils.draw import draw_squares
from utils.square import SquareDataset
from torch.utils.data import DataLoader

# make some pictures!
squares = SquareDataset(1280)
for i in range(5):
   print(squares[i])

#%%
device = torch.device('cuda')

# Use the nn package to define our model and loss function.
model = torch.nn.Linear(9, 3)
model = model.to(device)

cost = torch.nn.MSELoss(reduction='sum')

# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# dataset!
dataloader = DataLoader(squares, batch_size=128)

epochs = 300
#%% run training loop
for t in range(epochs):
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device) / 255, Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

    if t % 10 == 0:
        print('l: {:>8f}, (e {})'.format(loss.item(), t))

    
for p in model.parameters():
    print(p)
