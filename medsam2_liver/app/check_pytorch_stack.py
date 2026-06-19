#!/usr/bin/env python3
"""Check whether the Docker image has a usable PyTorch/CUDA stack.

Run inside the built container:

    docker run --rm --gpus all medsam2-liver:20ch python3 app/check_pytorch_stack.py --require-cuda

What this checks:
- Python version
- PyTorch import
- PyTorch version
- whether PyTorch was compiled with CUDA
- whether CUDA is visible at runtime
- cuDNN version
- a tiny GPU tensor operation, if CUDA is visible
"""

from __future__ import annotations

import argparse
import platform
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Check PyTorch/CUDA availability inside Docker")
    p.add_argument("--require-cuda", action="store_true", help="Exit nonzero if CUDA is not available")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("Python executable:", sys.executable)
    print("Python version   :", sys.version.replace("\n", " "))
    print("Platform         :", platform.platform())

    try:
        import torch
    except Exception as exc:
        print("ERROR: failed to import torch:", repr(exc))
        return 10

    print("PyTorch version  :", torch.__version__)
    print("Torch CUDA build :", torch.version.cuda)
    print("cuDNN version    :", torch.backends.cudnn.version())
    print("CUDA available   :", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print("Current device   :", idx)
        print("Device name      :", torch.cuda.get_device_name(idx))
        x = torch.ones((4, 4), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("GPU tensor test  : PASS", float(y.sum().item()))
    else:
        print("GPU tensor test  : SKIPPED")
        if args.require_cuda:
            print("ERROR: --require-cuda was set, but torch.cuda.is_available() is False")
            print("Common causes:")
            print("  - image contains CPU-only PyTorch")
            print("  - container was not run with --gpus all")
            print("  - NVIDIA Container Toolkit is not installed/configured")
            print("  - host NVIDIA driver is too old for the CUDA runtime in the image")
            return 20

    if torch.version.cuda is None and args.require_cuda:
        print("ERROR: PyTorch appears to be CPU-only because torch.version.cuda is None")
        return 30

    print("Stack check       : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
