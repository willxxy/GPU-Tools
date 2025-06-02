import torch
import sys

def check_pytorch_mps():
    print("PyTorch + MPS Installation Check")
    print("================================")

    # PyTorch build
    print(f"PyTorch version: {torch.__version__}")

    # MPS availability
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    print(f"MPS backend built:     {mps_built}")
    print(f"MPS device available:  {mps_available}")

    if mps_available:
        print("\nTensor Operations on MPS")
        print("------------------------")
        try:
            device = torch.device("mps")
            x = torch.tensor([1.0]).to(device)
            y = torch.tensor([2.0], device=device)
            z = x + y
            print(f"x device: {x.device} | value: {x}")
            print(f"y device: {y.device} | value: {y}")
            print(f"z = x + y -> {z} (device: {z.device})")
            print("\nSuccess: PyTorch can run on the Apple GPU (MPS)!")
        except Exception as e:
            print(f"\nError during MPS tensor operations: {e}")
    else:
        print("\nNote: MPS is unavailable. PyTorch will run on CPU.")

    # Basic system info
    print("\nSystem Information")
    print("------------------")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")

if __name__ == "__main__":
    check_pytorch_mps()
