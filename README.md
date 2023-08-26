# GPU Tools

Simple python tools for synchronizing PyTorch and GPU usage on respective OS.

Note: Mostly tested on torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113.

## Tools Available

1. Simple checks whether CUDA is available.
2. Identification of which GPU(s) are available and in use.
3. Simple checks for correctly allocating or creating tensor to and from GPU.
4. Simple checks for distributed training across multiple GPUs.
