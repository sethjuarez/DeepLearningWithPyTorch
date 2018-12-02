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
model = torch.nn.Sequential(
    torch.nn.Linear(9, 3)
)
model = model.to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)

# dataset!
dataloader = DataLoader(squares, batch_size=128)

epochs = 200
#%%
#LOOP
for t in range(epochs):
    for batch, (data, target) in enumerate(dataloader):
        data, target = data.to(device) / 255, target.to(device)
        pred = model(data)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if t % 10 == 0:
        print(t, loss.item())
    

for p in model.parameters():
    print(p)
