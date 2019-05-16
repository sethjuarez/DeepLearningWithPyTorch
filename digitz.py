import os
import torch
import argparse
import torch.nn as nn
from pathlib import Path
import torch.onnx as onnx
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

###################################################################
# Helpers                                                         #
###################################################################
def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def check_dir(path, check=False):
    if check:
        assert os.path.exists(path), '{} does not exist!'.format(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        return Path(path).resolve()

###################################################################
# Data Loader                                                     #
###################################################################
def get_dataloader(train=True, batch_size=64, data_dir='data'):
    digits = datasets.MNIST(data_dir, train=train, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.reshape(28*28))
                        ]),
                        target_transform=transforms.Compose([
                            transforms.Lambda(lambda y: 
                                torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
                        ])
                     )

    return DataLoader(digits, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

###################################################################
# Saving                                                          #
###################################################################
def save_model(model, device, path, name):
    base = Path(path)
    onnx_file = base.joinpath('{}.onnx'.format(name)).resolve()
    pth_file = base.joinpath('{}.pth'.format(name)).resolve()
    
    # create dummy variable to traverse graph
    x = torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255
    onnx.export(model, x, onnx_file)
    print('Saved onnx model to {}'.format(onnx_file))

    # saving PyTorch Model Dictionary
    torch.save(model.state_dict(), pth_file)
    print('Saved PyTorch Model to {}'.format(pth_file))

###################################################################
# Models                                                          #
###################################################################
class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.layer1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.layer1(x)
        return F.softmax(x, dim=1)

class NeuralNework(nn.Module):
    def __init__(self):
        super(NeuralNework, self).__init__()
        self.layer1 = nn.Linear(28*28, 512)
        self.layer2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return F.softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

###################################################################
# Train/Test                                                      #
###################################################################
def train(model, device, dataloader, cost, optimizer, epoch):
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print('loss: {:>10f}  [{:>5d}/{:>5d}]'.format(loss.item(), batch * len(X), len(dataloader.dataset)))

def test(model, device, dataloader, cost):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y.argmax(1)).type(torch.float).sum().item()

    test_loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)
    print('\nTest Error:')
    print('acc: {:>0.1f}%, avg loss: {:>8f}'.format(100*correct, test_loss))

###################################################################
# Main Loop                                                       #
###################################################################
def main(data_dir, output_dir, log_dir, epochs, batch, lr, model_kind):
    # use GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get data loaders
    training = get_dataloader(train=True, batch_size=batch, data_dir=data_dir)
    testing = get_dataloader(train=False, batch_size=batch, data_dir=data_dir)

    # model
    if model_kind == 'linear':
        model = Logistic().to(device)
    elif model_kind == 'nn':
        model = NeuralNework().to(device)
    else:
        model = CNN().to(device)

    info('Model')
    print(model)

    # cost function
    cost = torch.nn.BCELoss()

    # optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, 5)

    for epoch in range(1, epochs + 1):
        info('Epoch {}'.format(epoch))
        scheduler.step()
        print('Current learning rate: {}'.format(scheduler.get_lr()))
        train(model, device, training, cost, optimizer, epoch)
        test(model, device, testing, cost)

    # save model
    info('Saving Model')
    save_model(model, device, output_dir, 'model')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-o', '--output', help='output directory', default='outputs')
    parser.add_argument('-g', '--logs', help='log directory', default='logs')
    
    parser.add_argument('-e', '--epochs', help='number of epochs', default=15, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=100, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=0.001, type=float)

    parser.add_argument('-m', '--model', help='model type', default='cnn', choices=['linear', 'nn', 'cnn'])

    args = parser.parse_args()

    # enforce folder locatations
    args.data = check_dir(args.data).resolve()
    args.outputs = check_dir(args.output).resolve()
    args.logs = check_dir(args.logs).resolve()

    main(data_dir=args.data, output_dir=args.output, log_dir=args.logs, 
         epochs=args.epochs, batch=args.batch, lr=args.lr, model_kind=args.model)