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
# hvd 0
import horovod.torch as hvd

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
    # hvd 2
    if hvd.size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        shuffle = False
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Average)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    else:
        train_sampler = None
        shuffle = True
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=shuffle, num_workers=8*hvd.size(), pin_memory=True, sampler=train_sampler)
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
    # hvd 1
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    # hvd 2
    hvd.init()
    train(hvd.local_rank(), args)

if __name__ == '__main__':
    main()