"""
Model hashing.

Computes SHA256 hashes for ComfyUI model files and caches the result so a
model is only ever hashed once. The cache is a tree of per-model `.index`
JSON sidecar files under `model_index/`, keyed by the model's path *relative
to ComfyUI's models directory* (e.g. `checkpoints/SDXL/foo.safetensors`) so
that two models sharing a bare filename in different folders never collide.

It also keeps an in-memory snapshot of every known model's hash (`_INDEX`)
plus a cheap content version token (`_VERSION`). This is what the API layer
serves to the frontend in a single request — building the response is then
just a dict read, with no disk I/O per request.
"""
import os
import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Tuple

import folder_paths

# Per-model `.index` sidecar cache. Mirrors the models-dir tree so the files
# stay human-browsable. Gitignored — this is local user data.
INDEX_DIR = os.path.join(os.path.dirname(__file__), "model_index")

# Model folder types to scan on startup.
MODEL_TYPES = [
    "checkpoints", "loras", "vae", "embeddings", "upscale_models",
    "controlnet", "clip", "clip_vision", "diffusion_models",
]

# In-memory snapshot of every known model: cache_key -> index dict.
# Guarded by _LOCK because it is populated by the background startup thread
# while the API layer may read it from the aiohttp event loop.
_INDEX: Dict[str, Dict] = {}
_VERSION: str = "empty"
_LOCK = threading.Lock()


def cache_key(filepath: str) -> str:
    """
    Stable, collision-free cache key for a model file: its path relative to
    ComfyUI's models directory. Models living outside that directory (extra
    registered paths) fall back to their absolute path with leading separators
    stripped, which is still unique and still safe to use as a relative path.
    """
    abspath = os.path.abspath(filepath)
    rel = os.path.relpath(abspath, folder_paths.models_dir)
    if rel.startswith(".." + os.sep) or rel == "..":
        # Outside models_dir — use the absolute path, made relative-safe.
        rel = abspath.lstrip(os.sep)
        if os.name == "nt":  # strip a drive colon, e.g. C:\ -> C\
            rel = rel.replace(":", "", 1)
    return rel.replace(os.sep, "/")


def _index_path(key: str) -> str:
    """Filesystem path of the `.index` sidecar for a given cache key."""
    path = os.path.join(INDEX_DIR, *key.split("/")) + ".index"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def calculate_sha256(filepath: str) -> str:
    """Calculate the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def load_index(key: str) -> Optional[Dict]:
    """Return the cached index for a key, or None if it hasn't been hashed."""
    path = os.path.join(INDEX_DIR, *key.split("/")) + ".index"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_index(key: str, data: Dict) -> None:
    """Write the index sidecar for a key."""
    with open(_index_path(key), "w") as f:
        json.dump(data, f, indent=2)


def _compute_version(index: Dict[str, Dict]) -> str:
    """A cheap, stable content token over (key, hash) pairs — used as an ETag."""
    if not index:
        return "empty"
    digest = hashlib.sha256()
    for key in sorted(index):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update((index[key].get("hash") or "").encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _remember(entry: Dict) -> None:
    """Add/replace an entry in the in-memory index and refresh the version."""
    global _VERSION
    with _LOCK:
        _INDEX[entry["key"]] = entry
        _VERSION = _compute_version(_INDEX)


def hash_model(filepath: str, key: Optional[str] = None) -> Dict:
    """
    Ensure a model file is hashed. Returns its index dict. If the model has
    already been hashed (its `.index` sidecar exists), this is a cheap no-op.
    Either way the result is recorded in the in-memory index.
    """
    if key is None:
        key = cache_key(filepath)

    index = load_index(key)
    if not (index and index.get("hash")):
        print(f"[ImageLab] Hashing {key} ...")
        index = {
            "key": key,
            "filename": os.path.basename(filepath),
            "hash": calculate_sha256(filepath),
            "hashed_at": time.time(),
        }
        save_index(key, index)

    _remember(index)
    return index


def build_index() -> Dict[str, int]:
    """
    Scan every model folder, hash anything not yet hashed, and populate the
    in-memory index from the (now complete) on-disk cache. Run once at startup.
    Returns a per-type count of files newly hashed this run.
    """
    counts: Dict[str, int] = {}

    for model_type in MODEL_TYPES:
        if model_type not in folder_paths.folder_names_and_paths:
            continue

        for filename in folder_paths.get_filename_list(model_type):
            filepath = folder_paths.get_full_path(model_type, filename)
            if not filepath:
                continue
            key = cache_key(filepath)
            already_hashed = load_index(key) is not None
            try:
                hash_model(filepath, key)
                if not already_hashed:
                    counts[model_type] = counts.get(model_type, 0) + 1
            except Exception as e:
                print(f"[ImageLab] Error hashing {filename}: {e}")

    return counts


def get_snapshot() -> Tuple[str, List[Dict]]:
    """
    Return `(version, entries)` for the API layer. `version` is an ETag-style
    content token; `entries` is the list of `{key, filename, hash, hashed_at}`
    dicts. Cheap — a lock-guarded copy, no disk I/O.
    """
    with _LOCK:
        return _VERSION, list(_INDEX.values())


def forget(filepath: str) -> None:
    """
    Drop a model from the index: delete its `.index` sidecar and remove its
    in-memory entry. Called when a model file is deleted from disk. `cache_key`
    is pure path math, so this works whether or not the file still exists.
    """
    global _VERSION
    key = cache_key(filepath)
    path = os.path.join(INDEX_DIR, *key.split("/")) + ".index"
    if os.path.exists(path):
        os.remove(path)
    with _LOCK:
        if _INDEX.pop(key, None) is not None:
            _VERSION = _compute_version(_INDEX)
