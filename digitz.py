import torch
from utils.helpers import *
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
        pred = model(X)
        loss = cost(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
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
    print('acc: {:>8f}, loss: {:>8f}'.format(test_loss, 100*correct))

def main(epochs=10, learning_rate=.01):
    # use GPU
    device = torch.device('cuda')

    # get data loaders
    training = get_dataloader(train=True)
    testing = get_dataloader(train=False)

    # model
    model = Logistic().to(device)

    # cost function
    cost = torch.nn.MSELoss(reduction='sum')

    # optimizers
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)

    for epoch in range(1, epochs + 1):
        info('Epoch {}'.format(epoch))
        train(model, device, training, cost, optimizer, epoch)
        test(model, device, testing, cost)

    # tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hx))
    # tf.train.AdamOptimizer(settings.lr).minimize(cost)

if __name__ == '__main__':
    main()