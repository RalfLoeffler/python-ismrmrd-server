#!/usr/bin/env python3
"""Docker-friendly inference CLI for the MedSAM2 20-channel liver bundle.

Supported input formats
-----------------------
- .npz containing key ``echo_stack`` or ``image``
- .npy containing a 20-echo stack
- .mat containing key ``echo_stack`` or common alternatives

The expected image is a 20-channel echo stack. The script tries to orient any
axis of length 20 into channel-first shape [20, H, W].

Outputs
-------
A compressed .npz containing:
- prob_512
- pred_512
- logits_512
- prob_orig
- pred_orig
- input_shape
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    import h5py
except Exception:
    h5py = None

try:
    from scipy.io import loadmat as scipy_loadmat
except Exception:
    scipy_loadmat = None


DEFAULT_BUNDLE = Path("/opt/medsam2_liver/portable_medsam2_liver_bundle.pt")
DEFAULT_THRESHOLD = 0.5


def safe_torch_load(path: Path, *, map_location="cpu"):
    """Use weights_only=True when supported to avoid PyTorch FutureWarning."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def register_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("times"):
        OmegaConf.register_new_resolver("times", lambda x, y: float(x) * float(y))
    if not OmegaConf.has_resolver("divide"):
        OmegaConf.register_new_resolver("divide", lambda x, y: float(x) / float(y))


def resolve_existing_path(path_str: str, *, fallback_dirs: Sequence[Path]) -> str:
    """Resolve bundle paths that may have been created on Windows.

    If the path stored in the bundle is absolute and unavailable in Docker, this
    falls back to searching by filename in known Docker directories.
    """
    p = Path(path_str)
    if p.exists():
        return str(p)

    # Normalize Windows separators and try relative to current working dir.
    rel = Path(path_str.replace("\\", "/"))
    if rel.exists():
        return str(rel)

    for base in fallback_dirs:
        candidate = base / rel
        if candidate.exists():
            return str(candidate)
        candidate = base / rel.name
        if candidate.exists():
            return str(candidate)

    # Return original; build_sam2 will produce a clear error if unresolved.
    return path_str


def first_available_call(fn, candidate_calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]], label: str):
    last_exc = None
    for args, kwargs in candidate_calls:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last_exc = exc
    raise TypeError(f"Could not call {label} with any candidate signature. Last error: {last_exc}")


def build_model(device: torch.device, model_cfg: str, base_checkpoint: str):
    from sam2.build_sam import build_sam2_video_predictor

    register_omegaconf_resolvers()

    candidate_calls = [
        ((model_cfg, base_checkpoint, device), {}),
        ((model_cfg, base_checkpoint), {"device": device}),
        ((), {"config_file": model_cfg, "sam2_checkpoint": base_checkpoint, "device": device}),
        ((), {"model_cfg": model_cfg, "sam2_checkpoint": base_checkpoint, "device": device}),
        ((), {"config_name": model_cfg, "sam2_checkpoint": base_checkpoint, "device": device}),
    ]
    model = first_available_call(build_sam2_video_predictor, candidate_calls, "build_sam2_video_predictor")
    return model.to(device)


def encode_prompt(model, boxes: torch.Tensor):
    prompt_encoder = model.sam_prompt_encoder
    candidate_calls = [
        ((None,), {"boxes": boxes, "masks": None}),
        ((), {"points": None, "boxes": boxes, "masks": None}),
        ((None, boxes, None), {}),
        ((boxes,), {}),
        ((), {"boxes": boxes}),
    ]
    out = first_available_call(prompt_encoder, candidate_calls, "sam_prompt_encoder")
    if not (isinstance(out, tuple) and len(out) >= 2):
        raise RuntimeError(f"Unexpected prompt encoder output type: {type(out)}")
    sparse_embeddings = out[0]
    dense_embeddings = out[1]
    dense_pe = prompt_encoder.get_dense_pe() if hasattr(prompt_encoder, "get_dense_pe") else torch.zeros_like(dense_embeddings)
    return sparse_embeddings, dense_embeddings, dense_pe


def decode_masks(model, image_embed, image_pe, sparse_embeddings, dense_embeddings, high_res_feats):
    mask_decoder = model.sam_mask_decoder
    candidate_kwargs = [
        dict(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        ),
        dict(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_feats=high_res_feats,
        ),
    ]
    last_exc = None
    for kwargs in candidate_kwargs:
        try:
            return mask_decoder(**kwargs)
        except TypeError as exc:
            last_exc = exc
    raise TypeError(f"Could not call sam_mask_decoder with any candidate signature. Last error: {last_exc}")


def load_mat_any(path: Path) -> Dict[str, Any]:
    if h5py is not None:
        try:
            out: Dict[str, Any] = {}
            with h5py.File(path, "r") as f:
                for key in f.keys():
                    out[key] = np.array(f[key])
            return out
        except Exception:
            pass
    if scipy_loadmat is None:
        raise ImportError("Could not load .mat. Install scipy and h5py in the container.")
    data = scipy_loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in data.items() if not k.startswith("__")}


def extract_echo_stack(path: Path, key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        candidate_keys = [key] if key else ["echo_stack", "image", "images", "arr_0"]
        for k in candidate_keys:
            if k and k in data:
                return np.asarray(data[k], dtype=np.float32)
        raise KeyError(f"No echo stack key found in {path}. Keys: {list(data.keys())}")

    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)

    if suffix == ".mat":
        data = load_mat_any(path)
        candidate_keys = [key] if key else ["echo_stack", "image", "images", "LearnImage", "Test_Image"]
        for k in candidate_keys:
            if k and k in data:
                return np.asarray(data[k], dtype=np.float32)
        raise KeyError(f"No echo stack key found in {path}. Keys: {list(data.keys())}")

    raise ValueError(f"Unsupported input file type: {path.suffix}")


def orient_echo_stack(arr: np.ndarray) -> np.ndarray:
    """Return channel-first [20, H, W]."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D echo stack after squeeze, got shape {arr.shape}")

    axes_with_20 = [i for i, s in enumerate(arr.shape) if s == 20]
    if not axes_with_20:
        raise ValueError(f"No axis of length 20 found in echo stack shape {arr.shape}")

    echo_axis = axes_with_20[0]
    if echo_axis != 0:
        arr = np.moveaxis(arr, echo_axis, 0)

    if arr.shape[0] != 20:
        raise ValueError(f"Could not orient to [20,H,W], got {arr.shape}")

    return np.ascontiguousarray(arr, dtype=np.float32)


def predict_one(model, image_20hw: np.ndarray, *, device: torch.device, threshold: float, amp: bool, target_hw: Tuple[int, int]):
    original_hw = tuple(int(x) for x in image_20hw.shape[1:])
    image = torch.from_numpy(image_20hw).unsqueeze(0).to(device)  # [1,20,H,W]
    image_512 = F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32, device=device)

    amp_enabled = amp and device.type == "cuda"
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            features = model.forward_image(image_512)
            image_embed = features["backbone_fpn"][2]
            high_res_feats = [features["backbone_fpn"][0], features["backbone_fpn"][1]]
            sparse_embeddings, dense_embeddings, dense_pe = encode_prompt(model, boxes)
            decoder_out = decode_masks(
                model=model,
                image_embed=image_embed,
                image_pe=dense_pe,
                sparse_embeddings=sparse_embeddings,
                dense_embeddings=dense_embeddings,
                high_res_feats=high_res_feats,
            )
            low_res_logits = decoder_out[0]
            logits_512 = F.interpolate(low_res_logits, size=target_hw, mode="bilinear", align_corners=False)
        logits_512 = logits_512.float()
        prob_512 = torch.sigmoid(logits_512)
        pred_512 = (prob_512 >= threshold).to(torch.uint8)
        prob_orig = F.interpolate(prob_512, size=original_hw, mode="bilinear", align_corners=False)
        pred_orig = (prob_orig >= threshold).to(torch.uint8)

    return {
        "logits_512": logits_512.cpu().numpy()[0, 0].astype(np.float32),
        "prob_512": prob_512.cpu().numpy()[0, 0].astype(np.float32),
        "pred_512": pred_512.cpu().numpy()[0, 0].astype(np.uint8),
        "prob_orig": prob_orig.cpu().numpy()[0, 0].astype(np.float32),
        "pred_orig": pred_orig.cpu().numpy()[0, 0].astype(np.uint8),
        "input_shape": np.array(image_20hw.shape, dtype=np.int32),
        "original_hw": np.array(original_hw, dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedSAM2 20-channel liver inference")
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--input", type=Path, required=False, help="Input .npz/.npy/.mat echo stack")
    p.add_argument("--input-key", type=str, default=None, help="Optional key for .npz/.mat input")
    p.add_argument("--output", type=Path, default=Path("/outputs/prediction.npz"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--print-metadata", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = safe_torch_load(args.bundle, map_location="cpu")
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise TypeError("Bundle must be a dict containing state_dict")

    if args.print_metadata:
        meta = {k: v for k, v in bundle.items() if k != "state_dict"}
        print(json.dumps(meta, indent=2))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    fallback_dirs = [Path.cwd(), Path("/opt/medsam2_liver"), Path("/opt/medsam2_liver/checkpoints")]
    base_checkpoint = resolve_existing_path(bundle["base_checkpoint"], fallback_dirs=fallback_dirs)
    model_cfg = bundle["model_cfg"]
    target_hw = tuple(int(x) for x in bundle.get("target_hw", [512, 512]))

    print("Using device:", device)
    print("Model cfg   :", model_cfg)
    print("Base ckpt   :", base_checkpoint)
    print("Bundle      :", args.bundle)

    model = build_model(device, model_cfg, base_checkpoint)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    print("Model loaded.")

    if args.input is None:
        print("No --input supplied. Bundle/model load check completed.")
        return

    echo_stack = orient_echo_stack(extract_echo_stack(args.input, key=args.input_key))
    print("Input echo stack:", echo_stack.shape)
    result = predict_one(model, echo_stack, device=device, threshold=args.threshold, amp=args.amp, target_hw=target_hw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
