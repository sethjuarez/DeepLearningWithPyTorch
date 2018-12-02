import torch
from torch.utils.data.dataset import Dataset

class SquareDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.X = torch.randint(255, (size, 9), dtype=torch.float)

        real_w = torch.tensor([[1,1,1,0,0,0,0,0,0],
                               [0,0,0,1,1,1,0,0,0],
                               [0,0,0,0,0,0,1,1,1]], 
                               dtype=torch.float)

        self.Y = torch.argmax(self.X.mm(real_w.t()), 1)

    def __getitem__(self, index):
        y = torch.zeros(3, dtype=torch.float) \
                 .scatter_(0, self.Y[index], 1)
        return (self.X[index], y)

    def __len__(self):
        return self.size

if __name__ == "__main__":
    data = SquareDataset(10)
    for i in range(len(data)):
        print(data[i])