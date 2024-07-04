import os
import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from runners import train, test

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loaders(batch_size, rank, world_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def main(rank, world_size):
    setup(rank, world_size)
    
    model = SimpleCNN().to(rank)
    model = DDP(model, device_ids=[rank])

    train_loader, test_loader = get_data_loaders(batch_size=64, rank=rank, world_size=world_size)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, 11):
        train(rank, world_size, model, train_loader, optimizer, epoch)
        test(rank, model, test_loader)
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
