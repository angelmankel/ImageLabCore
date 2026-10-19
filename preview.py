"""
Live preview size.

ComfyUI shrinks every sampler preview to `--preview-size` pixels on the long side (default 512)
before it goes over the websocket. The full-size picture already exists at that point: TAESD decodes
the latent at the render's resolution and the cap only throws pixels away. The cap is a module
global in ComfyUI's `latent_preview`, read on every preview, so it can be changed in a running
server. This module sets it at import (env `IMAGELAB_PREVIEW_SIZE`, otherwise at least
`DEFAULT`) and lets `/imagelab/api/preview-size` move it later with no restart.

Each ComfyUI process has its own cap: the pod's second (video) ComfyUI reads this module too, but
nginx sends `/video-comfy/imagelab/api/preview-size` to the first one, so the second keeps whatever
it had at start.
"""
import os

import latent_preview

DEFAULT = 1024
MIN, MAX = 64, 8192
# "No cap": larger than any render, so PIL's thumbnail() never resizes. Not sys.maxsize, which
# would overflow inside PIL.
UNLIMITED = 1 << 15


def get() -> int:
    return int(latent_preview.MAX_PREVIEW_RESOLUTION)


def is_unlimited(size: int) -> bool:
    return size >= UNLIMITED


def set_size(value) -> int:
    """Set the cap. 0, "max", "none" or "unlimited" lift it. Returns what was set. Raises ValueError."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("max", "none", "unlimited", "off"):
            value = 0
        else:
            value = int(v)
    size = int(value)
    if size == 0:
        size = UNLIMITED
    elif not (MIN <= size <= MAX):
        raise ValueError(f"size must be 0 (no cap) or {MIN}..{MAX}")
    latent_preview.MAX_PREVIEW_RESOLUTION = size
    return size


def apply_env() -> int:
    """At import: `IMAGELAB_PREVIEW_SIZE` wins; otherwise never go below DEFAULT, but keep a larger
    `--preview-size` someone passed on purpose."""
    raw = os.environ.get("IMAGELAB_PREVIEW_SIZE", "").strip()
    try:
        if raw:
            return set_size(raw)
        return set_size(max(get(), DEFAULT))
    except ValueError as e:
        print(f"[ImageLab] IMAGELAB_PREVIEW_SIZE={raw!r} ignored: {e}")
        return get()
