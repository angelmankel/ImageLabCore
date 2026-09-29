"""
Put model files where ComfyUI's loaders look for them.

CivitAI calls every base model a "Checkpoint", so a download used to land in `checkpoints/` even
when the file is a diffusion model only (Wan, Flux, Z-Image, Qwen-Image, Anima…). Those have no
VAE and no text encoder inside, so `CheckpointLoaderSimple` cannot load them, and `UNETLoader`
only lists `diffusion_models/`. Such a file is unusable where it lands.

A safetensors file says what it holds in its header, which is a few KB of JSON at the start.
A full checkpoint carries text-encoder weights under well-known prefixes. A file without any is not
a usable checkpoint — `CheckpointLoaderSimple` would hand back no CLIP, so no prompt could be
encoded — and moves to `diffusion_models/`, where `UNETLoader` reads it (a bundled VAE is just
ignored there). The move is a rename on the
same disk, so the hash index is carried over rather than recomputed (`hashing.relocate`).

Runs after each download into `checkpoints/`, once at startup over the whole folder, and on
`POST /imagelab/api/models/sort`.
"""
import json
import os
import struct
from typing import Dict, List, Optional

import folder_paths

from . import hashing

# Weight-name prefixes of the parts a full checkpoint bundles besides the diffusion model.
_VAE_PREFIXES = ("first_stage_model.", "vae.")
_TEXT_PREFIXES = ("cond_stage_model.", "conditioner.", "text_encoders.", "text_encoder.", "clip.")
# A header larger than this is not a safetensors header.
_MAX_HEADER = 100 * 1024 * 1024


def safetensors_keys(path: str) -> Optional[List[str]]:
    """The tensor names in a safetensors file, or None when it is not one (or unreadable)."""
    if not path.endswith(".safetensors"):
        return None
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                return None
            (size,) = struct.unpack("<Q", raw)
            if size <= 0 or size > _MAX_HEADER:
                return None
            header = json.loads(fh.read(size))
    except (OSError, ValueError):
        return None
    if not isinstance(header, dict):
        return None
    return [k for k in header if k != "__metadata__"]


def is_diffusion_only(keys: List[str]) -> bool:
    """True when the weights hold no text encoder: a diffusion model, perhaps with its VAE."""
    if not keys:
        return False
    return not any(k.startswith(_TEXT_PREFIXES) for k in keys)


def describe(path: str) -> Optional[Dict]:
    """What a safetensors file holds: tensor count, counts per top-level name prefix, and whether
    it has a VAE / text encoder. None when it is not a safetensors file."""
    keys = safetensors_keys(path)
    if keys is None:
        return None
    prefixes: Dict[str, int] = {}
    for k in keys:
        head = k.split(".", 2)
        p = ".".join(head[:2]) if head[0] in ("model", "text_encoders") and len(head) > 2 else head[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    return {
        "tensors": len(keys),
        "prefixes": dict(sorted(prefixes.items(), key=lambda kv: -kv[1])[:40]),
        "has_vae": any(k.startswith(_VAE_PREFIXES) for k in keys),
        "has_text_encoder": any(k.startswith(_TEXT_PREFIXES) for k in keys),
        "diffusion_only": is_diffusion_only(keys),
    }


def _folder_dir(name: str) -> Optional[str]:
    entry = folder_paths.folder_names_and_paths.get(name)
    if not entry or not entry[0]:
        return None
    paths = entry[0]
    return paths[0] if isinstance(paths, list) else paths


def diffusion_models_dir() -> Optional[str]:
    """Where UNETLoader looks first. Older ComfyUI calls the folder `unet`."""
    return _folder_dir("diffusion_models") or _folder_dir("unet")


def sort_file(path: str) -> Optional[str]:
    """
    If `path` (in checkpoints/) is a diffusion model only, move it to diffusion_models/ and
    return the new path. Otherwise leave it and return None. Never overwrites a file.
    """
    keys = safetensors_keys(path)
    if keys is None or not is_diffusion_only(keys):
        return None
    target_dir = diffusion_models_dir()
    if not target_dir:
        return None
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, os.path.basename(path))
    if os.path.exists(target):
        print(f"[ImageLab] {os.path.basename(path)} is a diffusion model, but diffusion_models/ already has one by that name — left in checkpoints/")
        return None
    os.replace(path, target)
    hashing.relocate(path, target)
    print(f"[ImageLab] Moved {os.path.basename(path)} to diffusion_models/ (no VAE or text encoder inside)")
    return target


def sort_checkpoints() -> Dict[str, List[str]]:
    """Move every diffusion-model-only file out of checkpoints/. Returns `{moved: [names]}`."""
    moved: List[str] = []
    for name in folder_paths.get_filename_list("checkpoints"):
        path = folder_paths.get_full_path("checkpoints", name)
        if not path or os.path.exists(path + ".aria2") or not os.path.isfile(path):
            continue
        try:
            if sort_file(path):
                moved.append(name)
        except OSError as e:
            print(f"[ImageLab] Could not sort {name}: {e}")
    return {"moved": moved}
