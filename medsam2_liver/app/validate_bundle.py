#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def safe_torch_load(path: Path, *, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    p = argparse.ArgumentParser(description="Validate MedSAM2 liver bundle metadata")
    p.add_argument("--bundle", type=Path, default=Path("portable_medsam2_liver_bundle.pt"))
    args = p.parse_args()
    obj = safe_torch_load(args.bundle, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("Bundle is not a dict")
    required = ["state_dict", "model_cfg", "prompt_mode", "target_hw", "base_checkpoint"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f"Missing bundle keys: {missing}")
    meta = {k: v for k, v in obj.items() if k != "state_dict"}
    print(json.dumps(meta, indent=2))
    print("state_dict tensors:", len(obj["state_dict"]))
    print("Bundle metadata looks valid.")


if __name__ == "__main__":
    main()
