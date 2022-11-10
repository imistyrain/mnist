#https://baijiahao.baidu.com/s?id=1707873363595994690&wfr=spider&for=pc
import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
# from apex import amp
from model import ConvNet

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int)
    parser.add_argument('-g', '--gpus', default=1, type=int)
    parser.add_argument('-lr', '--lr', default=1e-2, type=float)
    parser.add_argument('-nr', '--nr', default=0, type=int)
    parser.add_argument('-es', '--embedding_size', default=2, type=int)
    parser.add_argument('-bs', '--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()
    return args

def eval(model):
    test_dataset = torchvision.datasets.MNIST( root='./data', train=False, transform=transforms.ToTensor(),download=True)   
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            acc = 100 * correct / total
        return acc

def train(gpu, args):
    rank = 0
    if args.world_size > 1:
        rank = args.nr * args.gpus + gpu	                          
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)                              
    torch.manual_seed(0)
    model = ConvNet(args.embedding_size)
    torch.cuda.set_device(gpu)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=args.world_size,rank=rank)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=False,num_workers=0,pin_memory=True,sampler=train_sampler) 
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=int(((len(train_dataset)-1)//args.batch_size + 1)*args.epochs), cycle_momentum=False)    
    start = datetime.now()
    total_step = len(train_loader)
    best_acc = 0
    correct = 0
    total = 0
    for epoch in range(args.epochs):
        if rank == 0:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        model.train()
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
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
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
            scheduler.step()
            msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch, args.epochs, i + 1,  total_step, loss.item(), acc)
            if rank == 0:
                pbar.set_description(msg)      
        eval_acc = eval(model)
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

def main():
    args = get_args()
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    args.world_size = args.gpus * args.nodes                #
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'              #
        os.environ['MASTER_PORT'] = '8889'                      #
        mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    else:
        train(0, args)

if __name__=="__main__":
    main()