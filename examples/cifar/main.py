# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Train CIFAR10 / CIFAR100 with PyTorch."""
import os
import argparse
import math
import time

import torch
import torch.distributed as torch_ddp
import torchvision

from torch import optim
from torch import nn
from torchvision import models
from torchvision import transforms

import periflow_sdk as pf


def print_rank_0(msg):
    if torch_ddp.get_rank() != 0:
        return

    print(msg, flush=True)


def train_step(inputs,
               labels,
               model,
               loss_function,
               optimizer,
               lr_scheduler):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()

    optimizer.step()
    lr_scheduler.step()

    return loss.item(), lr_scheduler.get_last_lr()[0]


def validation(test_loader, model, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(model.device)
        labels = labels.to(model.device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            _, predicted = outputs.max(1)
            test_loss += loss * labels.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return test_loss / total, correct / total * 100


def main(args):
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        # cifar 100
        num_classes = 100

    if args.use_cpu:
        device_ids = None
        output_device = None
    else:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device_ids = [torch.cuda.current_device()]
        output_device = torch.cuda.current_device()

    world_size = torch_ddp.get_world_size()

    net = getattr(models, args.model)(num_classes=num_classes)
    if not args.use_cpu:
        net.cuda(output_device)

    net = nn.parallel.DistributedDataParallel(net, device_ids=device_ids, output_device=output_device)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(36),  # CIFAR is 32, but for compatibility with ImageNet model
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = getattr(torchvision.datasets, args.dataset.upper())(
        root=args.data_path, train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=torch.utils.data.distributed.DistributedSampler(train_dataset),
                                               num_workers=args.num_dataloader_workers)

    transform_test = transforms.Compose([
        transforms.Resize(36),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_dataset = getattr(torchvision.datasets, args.dataset.upper())(
        root=args.data_path, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_dataloader_workers)

    args.batch_size = args.batch_size * world_size
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_steps = args.total_epochs * steps_per_epoch
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100 * steps_per_epoch, 200 * steps_per_epoch],
                                                        gamma=0.1)

    if args.load and os.path.exists(os.path.join(args.load, "checkpoint.pt")):
        ckpt = torch.load(os.path.join(args.load, "checkpoint.pt"), map_location="cpu")
        step = ckpt["latest_step"]
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        epoch = math.ceil(step / steps_per_epoch)
    else:
        step = 0
        epoch = 1

    pf.init(total_train_steps=total_steps)

    # now train
    train_iterator = iter(train_loader)
    net.train()
    start_time = time.time()
    while step < total_steps:
        try:
            inputs, labels = next(train_iterator)
            inputs = inputs.to(net.device)
            labels = labels.to(net.device)
        except StopIteration:
            optimizer.zero_grad()
            epoch += 1
            train_iterator = iter(train_loader)
            continue

        with pf.train_step():
            loss, learning_rate = train_step(inputs=inputs,
                                             labels=labels,
                                             model=net,
                                             loss_function=loss_function,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler)
            if not args.use_cpu:
                torch.cuda.synchronize()
            end_time = time.time()

        throughput = (inputs.size(0) * world_size) / (end_time - start_time)
        step += 1

        pf.metric({
            "iteration": step,
            "loss": loss,
            "learning_rate": learning_rate
        })

        if step % args.log_interval == 0:
            print_rank_0("[Step %d / %d (%.1f %%)] Training Loss = %.5f, Learning Rate = %.5f Throughput = %.2f images / sec" % \
                         (step, total_steps, step / total_steps * 100, loss, learning_rate, throughput))

        if args.save and (step % args.save_interval == 0 or pf.is_emergency_save()):
            if torch_ddp.get_rank() == 0:
                torch.save({"latest_step": step,
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()},
                           os.path.join(args.save, "checkpoint.pt"))

            pf.upload_checkpoint()

        if step % args.test_interval == 0:
            loss, acc = validation(test_loader, net, loss_function)
            print_rank_0(f"Validation at step {step}: Loss = {loss} Accuracy = {acc}")
            net.train()

        start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--total-epochs', default=200, type=int, help='The total number of epochs')
    parser.add_argument('--batch-size', default=256, type=int, help='The default batch size')
    parser.add_argument('--save-interval', default=500, type=int, help='The checkpoint save intervals')
    parser.add_argument('--log-interval', default=1000, type=int, help='The logging intervals')
    parser.add_argument('--test-interval', default=500, type=int, help='Validation intervals')
    parser.add_argument('--save', default=None, type=str, help='The path to the save directory')
    parser.add_argument('--load', default=None, type=str, help='The path to the load directory')
    parser.add_argument('--num-dataloader-workers', default=4, type=int, help='The number of data loading processes')
    parser.add_argument('--use-cpu', default=False, action='store_true', help='whether training on cpu')
    parser.add_argument('--model', default='resnet50', help='model to use')
    parser.add_argument('--dataset', default='cifar10', help='model to use', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data-path', default='.', help='dataset path')
    args = parser.parse_args()

    if args.use_cpu:
        backend = 'gloo'
    else:
        assert torch.cuda.is_available()
        backend = 'nccl'

    torch_ddp.init_process_group(backend=backend)
    main(args)
