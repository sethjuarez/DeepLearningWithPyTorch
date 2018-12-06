import torch
import torch.onnx as onnx
import torch.optim as optim
from utils.helpers import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Logistic, NeuralNework, CNN

def get_dataloader(train=True, batch_size=64):
    digits = datasets.MNIST('data', train=train, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.reshape(28*28))
                        ]),
                        target_transform=transforms.Compose([
                            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, y, 1))
                        ])
                     )

    return DataLoader(digits, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


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

def save_model(model, device, path):
    # create dummy variable to traverse graph
    x = Variable(torch.randint(255, (1, 28*28), dtype=torch.float).to(device) / 255)
    onnx.export(model, x, path)
    print('Saved model to {}'.format(path))


def main(epochs=5, learning_rate=1e-3):
    # use GPU
    device = torch.device('cuda')

    # get data loaders
    training = get_dataloader(train=True)
    testing = get_dataloader(train=False)

    # model
    model = CNN().to(device)
    info('Model')
    print(model)


    # cost function
    cost = torch.nn.BCELoss()

    # optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        info('Epoch {}'.format(epoch))
        train(model, device, training, cost, optimizer, epoch)
        test(model, device, testing, cost)

    # save model
    info('Saving Model')
    save_model(model, device, 'model.onnx')
    print('Saving PyTorch Model as model.pth')
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()