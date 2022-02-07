'''Train CIFAR10 with PyTorch.'''
from dataclasses import dataclass
import os
import argparse
import math
from pathlib import Path
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import VGG

from periflow_sdk import periflow as pf
from periflow_sdk.dataloading.sampler import ResumableRandomSampler

def train_batch(inputs,
                targets,
                iteration,
                model,
                optimizer,
                lr_scheduler):

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    pf.log({"training_loss": loss.item(),
            "learning_rate": lr_scheduler.get_last_lr()})

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.*correct/total


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--total-steps', '-s', default=58800, type=int, help='The total number of training steps')
parser.add_argument('--batch-size', '-b', default=256, type=int, help='The default batch size')
parser.add_argument('--save-interval', '-i', default=500, type=int, help='The checkpoint save intervals')
parser.add_argument('--save-dir', '-dir', default='save', type=str, help='The path to the save directory')
parser.add_argument('--local-rank', '-r', default=0, type=int, help='The local rank of this process')
parser.add_argument('--num-workers', '-w', default=4, type=int, help='The number of data loading processes')
parser.add_argument('--seed', default=777, type=int, help='The seed for random generator')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG16')
net = net.to(device)
world_size = 0 if 'WORLD_SIZE' not in os.environ else int(os.environ['WORLD_SIZE'])
is_ddp = world_size > 1

if is_ddp:
    download = args.local_rank == 0
else:
    download = True

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=download, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

if is_ddp:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    net.cuda(args.local_rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

elif device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if is_ddp:
    # Use distributed sampler
    sampler = ResumableRandomSampler(len(trainset),
                                     args.batch_size,
                                     False,
                                     77,
                                     args.local_rank,
                                     world_size)
    download = args.local_rank == 0
else:
    sampler = ResumableRandomSampler(len(trainset), args.batch_size, False)
    download = True

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=sampler,
                                          num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=args.num_workers)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
steps_per_epoch = math.ceil(len(trainset) / args.batch_size)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[100 * steps_per_epoch, 200 * steps_per_epoch],
                                                 gamma=0.1)

trainloader_iter = iter(trainloader)

net.train()

ckpt_path = Path(args.save_dir) / "checkpoint.pt"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path)
    latest_step = ckpt['latest_step']
    net.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['lr_scheduler'])
    sampler.set_processed_steps(latest_step)
else:
    latest_step = 0

pf.init(args.total_steps,
        processed_steps=latest_step,
        local_rank=args.local_rank)


epoch = latest_step * args.batch_size // len(trainset) + 1

for step in range(latest_step + 1, args.total_steps + 1):
    try:
        inputs, targets = next(trainloader_iter)
        inputs = inputs.to(device)
        targets = targets.to(device)
    except StopIteration:
        # This indicates an end of epoch.
        acc = test()
        print(f"Epoch {epoch} has finished! Accuracy = {acc}")
        net.train()
        epoch += 1

        trainloader_iter = iter(trainloader)
        inputs, targets = next(trainloader_iter)
        inputs = inputs.to(device)
        targets = targets.to(device)
    with pf.train_step():
        # Actual training function
        train_batch(inputs=inputs,
                    targets=targets,
                    iteration=step,
                    model=net,
                    optimizer=optimizer,
                    lr_scheduler=scheduler)
        if step % args.save_interval == 0:
            pf.save({'latest_step': step,
                     'model': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'lr_scheduler': scheduler.state_dict()}, ckpt_path)
