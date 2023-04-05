import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

os.environ['MASTER_ADDR'] = ''  
os.environ['MASTER_PORT'] = '' 

def setup_process(rank, world_size):
    """Initialize the distributed process."""
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """Cleanup the distributed process."""
    dist.destroy_process_group()

def run(rank, world_size):
    """Run a distributed training process."""
    setup_process(rank, world_size)
    # Only rank 0 process will print the message
    if rank == 0:
        print("Distributed training is working correctly!")
    cleanup()

if __name__ == '__main__':
    world_size = 1  # Specify the number of processes
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)