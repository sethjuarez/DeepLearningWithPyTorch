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

        y = torch.argmax(self.X.mm(real_w.t()), 1)
        
        self.Y = torch.zeros(size, 3, dtype=torch.float) \
                      .scatter_(1, y.view(-1, 1), 1)

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

    def __len__(self):
        return self.size

if __name__ == "__main__":
    data = SquareDataset(10)
    for i in range(len(data)):
        print(data[i])



        