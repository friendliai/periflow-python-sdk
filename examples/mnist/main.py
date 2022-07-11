import os
import math
import argparse

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import distributed as torch_ddp

from torchvision import datasets
from torchvision.transforms import ToTensor

import periflow_sdk as pf


class CNNClassifier(nn.Module):
    """ A simple classifier model implemented by CNN """

    num_classes = 10

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(args: argparse.Namespace):

    # Device setup
    if args.use_cpu:
        backend = 'gloo'
    else:
        assert torch.cuda.is_available()
        backend = 'nccl'
    torch_ddp.init_process_group(backend=backend)
    world_size = torch_ddp.get_world_size()

    if args.use_cpu:
        device = None
        print_once("Using CPU")
    else:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.cuda.current_device()
        print_once("Using GPU")

    # Prepare dataset
    train_data = datasets.MNIST(
        root='/workspace/mnist/data', train=True, transform=ToTensor(), download=False
    )
    test_data = datasets.MNIST(
        root='/workspace/mnist/data', train=False, transform=ToTensor(), download=False
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Prepare model
    model = CNNClassifier()
    if not args.use_cpu:
        model.cuda(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    args.batch_size *= world_size
    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    
    # Load starting checkpoint
    if args.load and os.path.exists(os.path.join(args.load, "checkpoint.pt")):
        ckpt = torch.load(os.path.join(args.load, "checkpoint.pt"), map_location="cpu")
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['latest_step']
        epoch = math.ceil(step / steps_per_epoch)
    else:
        step = 0
        epoch = 1

    # Initialize Periflow
    pf.init(total_train_steps=total_steps)

    # Train
    train_itr = iter(train_loader)
    model.train()

    while step < total_steps:

        try:
            inputs, labels = next(train_itr)
            # inputs : [batch_size, 1, 28, 28]
            # labels : [batch_size, 10]
            inputs = inputs.to(device)
            labels = labels.to(device)
        except StopIteration:
            # when an epoch has ben finished
            optimizer.zero_grad()
            epoch += 1
            train_itr = iter(train_loader)
            continue

        # A training step
        with pf.train_step():
            loss = train_step(inputs, labels, model, optimizer, criterion)
            if not args.use_cpu:
                torch.cuda.synchronize()
        step += 1

        # Log training loss
        pf.metric({
            "iteration": step,
            "loss": loss
        })
        if step % args.log_interval == 0:
            print_once(f"[Epoch {epoch} Step {step}/{total_steps} loss={loss:.5f}")
        
        # Save checkpoint
        if args.save and (step % args.save_interval == 0 or pf.is_emergency_save()):
            if torch_ddp.get_rank() == 0:
                torch.save({
                    "latest_step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, os.path.join(args.save, "checkpoint.pt"))

            pf.upload_checkpoint()      # let Periflow know a checkpoint has just been saved
        
        # Validation
        if step % args.val_interval == 0:
            val_loss, acc = validate(test_loader, model, criterion)
            print_once(f"Validation at step {step} : loss={val_loss:.5f}, acc={acc:.3f}%")
            

def train_step(inputs, labels, model, optimizer, loss_func):
    """ Implements a single training step """
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(loader, model, loss_func):
    """ Validates model with given test dataset """
    model.eval()
    test_loss = 0
    total_size = 0
    correct = 0

    device = model.device

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            result = torch.max(outputs, dim=-1).indices
            
            test_loss += loss * labels.size(0)
            total_size += labels.size(0)
            correct += torch.eq(result, labels).sum().item()
        
    test_loss /= total_size
    acc = correct / total_size * 100
        
    model.train()
    return test_loss, acc


def print_once(msg):
    """ Prints a message only in the main process """
    if torch_ddp.get_rank() != 0:
        return
    print(msg, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch MNIST Training")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='The total number of epochs')
    parser.add_argument('--batch-size', default=256, type=int, help='The default batch size')
    parser.add_argument('--save-interval', default=500, type=int, help='The checkpoint save intervals')
    parser.add_argument('--log-interval', default=1000, type=int, help='The logging intervals')
    parser.add_argument('--val-interval', default=500, type=int, help='Validation intervals')
    parser.add_argument('--use-cpu', default=False, action='store_true', help='whether training on cpu')
    parser.add_argument('--save', default=None, type=str, help='The path to the save directory')
    parser.add_argument('--load', default=None, type=str, help='The path to the load directory')
    args = parser.parse_args()

    main(args)
