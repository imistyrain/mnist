import os
import torch
import torchvision
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from datetime import datetime
from tqdm import tqdm

from model import ConvNet
from util import eval

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(gpu, args):
    rank = 0
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True, num_workers=8, pin_memory=True)
    best_acc = 0
    correct = 0
    total = 0
    start = datetime.now()
    for epoch in range(args.epochs):
        if rank == 0:
            print("Epoch: {}/{}".format(epoch+1, args.epochs))
        pbar = tqdm(train_loader)
        model.train()
        for images, labels in pbar:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            acc = 100 * correct / total
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            msg = 'Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch, args.epochs, loss.item(), acc)
            if rank == 0:
                pbar.set_description(msg)
        eval_acc = eval(model, device)
        if rank == 0:
            print('Epoch {} test_acc: {} %'.format(epoch, eval_acc))
            if eval_acc > best_acc:
                best_acc = eval_acc
                print("best to {:.2f}".format(best_acc))
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(model.state_dict(), 'models/best.pth')

    if rank == 0:
        print("Training complete in: " + str(datetime.now() - start))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--lr', default=1e-2, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    train(0, args)

if __name__ == '__main__':
    main()