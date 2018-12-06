#%%
import torch
from utils.draw import draw_digits
from utils.square import SquareDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# make some pictures!
digits = datasets.MNIST('data', download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.reshape(28*28))
                        ]),
                        target_transform=transforms.Compose([
                            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, y, 1))
                        ])
                     )

#%%
print(digits[0])
draw_digits(digits)

#%%
device = torch.device('cuda')

# Use the nn package to define our model and loss function.
model = torch.nn.Linear(28*28, 10)
model = model.to(device)

cost = torch.nn.MSELoss(reduction='sum')

# optimizer which Tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# dataset!
dataloader = DataLoader(digits, batch_size=64, num_workers=0, pin_memory=True)

epochs = 10
#%% run training loop
for t in range(epochs):
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

    print('l: {:>8f}, (e {})'.format(loss.item(), t))
    
for p in model.parameters():
    print(p)
