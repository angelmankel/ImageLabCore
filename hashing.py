"""
Model hashing.

Computes SHA256 hashes for ComfyUI model files and caches the result so an
unchanged model is only hashed once. The cache is a tree of per-model `.index`
JSON sidecar files under `model_index/`, keyed by the model's path *relative
to ComfyUI's models directory* (e.g. `checkpoints/SDXL/foo.safetensors`) so
that two models sharing a bare filename in different folders never collide.
A folder registered outside the models directory (extra_model_paths.yaml) is
keyed `<type>/<path below that folder>`, never by its absolute path, so moving
the files or swapping the yaml for a symlink keeps (or migrates) every key.

It also keeps an in-memory snapshot of every known model's hash (`_INDEX`)
plus a cheap content version token (`_VERSION`). This is what the API layer
serves to the frontend in a single request — building the response is then
just a dict read, with no disk I/O per request.

Hashing never hides a model that is already known:
  - `preload()` publishes every sidecar on disk before anything else runs
    (local JSON reads only, no model file is touched);
  - a scan first reconciles every model file with the sidecars (a file whose
    key changed — moved, or the models folder became a symlink — takes over
    the sidecar with the same file name + size + mtime_ns, which is copied to
    the new key; old sidecars are never deleted), then hashes what is left,
    smallest file first, publishing each hash as soon as it exists;
  - models without a hash yet are listed in `get_status()["pending"]`.
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

# Model folder types to scan.
MODEL_TYPES = [
    "checkpoints", "loras", "vae", "embeddings", "upscale_models",
    "controlnet", "clip", "clip_vision", "diffusion_models",
    "unet_gguf",  # ComfyUI-GGUF: the .gguf files in diffusion_models/
]

# In-memory snapshot of every known model: cache_key -> index dict.
# Guarded by _LOCK because it is populated by the background startup thread
# while the API layer may read it from the aiohttp event loop.
_INDEX: Dict[str, Dict] = {}
# Models found on disk without a usable hash yet: cache_key -> {key, filename,
# folder, size, state: "queued" | "hashing" | "error", error?}.
_PENDING: Dict[str, Dict] = {}
# False until the first scan has listed every model file. Before that `_INDEX`
# holds the preloaded sidecars and `_PENDING` is not known yet.
_READY = False
_VERSION: str = "empty"
_LOCK = threading.Lock()
# One lock per key: downloads and the scanner can discover the same file at
# once, but a download finishing must not wait behind a 6 GB file being read
# over the network.
_KEY_LOCKS: Dict[str, threading.Lock] = {}
# The file being read right now: {key, filename, size, done} (bytes).
_CURRENT: Dict = {}


def _file_signature(filepath: str) -> Dict:
    stat = os.stat(filepath)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _under(path: str, root: str) -> Optional[str]:
    """`path` relative to `root` when it lies below it, else None. Case-insensitive on Windows."""
    root = root.rstrip("/\\")
    if not root:
        return None
    p, r = os.path.normcase(path), os.path.normcase(root)
    if len(p) > len(r) + 1 and p.startswith(r) and p[len(r)] in ("/", os.sep):
        return path[len(root) + 1:]
    return None


def _folder_roots() -> List[Tuple[str, str]]:
    """
    `(root dir, key prefix)` for every registered model folder, deepest first. A folder below
    the models dir keeps its path there as the prefix (`checkpoints`, `diffusion_models`); one
    outside it (extra_model_paths.yaml) is prefixed with its folder type. Each root is listed both
    as given and with symlinks resolved, so a path that comes back resolved still gets the same key.
    """
    models = os.path.abspath(folder_paths.models_dir)
    names = list(MODEL_TYPES) + [n for n in getattr(folder_paths, "folder_names_and_paths", {}) if n not in MODEL_TYPES]
    roots: List[Tuple[str, str]] = []
    for name in names:
        entry = folder_paths.folder_names_and_paths.get(name)
        paths = entry[0] if entry and isinstance(entry[0], (list, tuple)) else []
        prefix_name = "diffusion_models" if name == "unet_gguf" else name
        for base in paths:
            absbase = os.path.abspath(base)
            inside = _under(absbase, models)
            prefix = inside.replace(os.sep, "/") if inside is not None else prefix_name
            for root in {absbase, os.path.realpath(absbase)}:
                roots.append((root, prefix))
    realmodels = os.path.realpath(models)
    if realmodels != models:
        roots.append((realmodels, ""))
    roots.sort(key=lambda r: len(r[0]), reverse=True)  # stable: earlier types win ties
    return roots


def cache_key(filepath: str) -> str:
    """
    Stable, collision-free cache key for a model file: its path relative to
    ComfyUI's models directory. The path is not resolved, so a symlinked
    `models/checkpoints` still gives `checkpoints/<file>`. A model in a folder
    registered outside the models directory is `<type>/<path below that folder>`;
    only a file below no registered folder falls back to its absolute path with
    leading separators stripped (still unique, still a safe relative path).
    """
    abspath = os.path.abspath(filepath)
    rel = _under(abspath, os.path.abspath(folder_paths.models_dir))
    if rel is None:
        try:
            roots = _folder_roots()
        except Exception:
            roots = []
        for root, prefix in roots:
            below = _under(abspath, root)
            if below is not None:
                rel = os.path.join(prefix, below) if prefix else below
                break
    if rel is None:
        # Below no registered folder — use the absolute path, made relative-safe.
        rel = abspath.lstrip("/\\")
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
    track = _CURRENT.get("path") == filepath
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)
            if track:
                _CURRENT["done"] = _CURRENT.get("done", 0) + len(block)
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


def _valid(entry: Optional[Dict]) -> bool:
    """A sidecar that can be trusted for a file whose signature we compare against it."""
    return bool(isinstance(entry, dict) and entry.get("hash") and isinstance(entry.get("file"), dict))


def _load_sidecars() -> Dict[str, Dict]:
    """Every readable sidecar on disk, keyed by its path below INDEX_DIR (the key it is filed under)."""
    found: Dict[str, Dict] = {}
    if not os.path.isdir(INDEX_DIR):
        return found
    for folder, _dirs, files in os.walk(INDEX_DIR):
        for name in files:
            if not name.endswith(".index"):
                continue
            path = os.path.join(folder, name)
            key = os.path.relpath(path, INDEX_DIR)[: -len(".index")].replace(os.sep, "/")
            try:
                with open(path, "r") as f:
                    entry = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if isinstance(entry, dict):
                # The file's location is the key; a copied sidecar may still carry its old "key".
                found[key] = dict(entry, key=key)
    return found


def _compute_version(index: Dict[str, Dict], pending: Optional[Dict[str, Dict]] = None, ready: bool = True) -> str:
    """A cheap, stable content token over (key, hash) pairs and the pending list — used as an ETag."""
    if not index and not pending and ready:
        return "empty"
    digest = hashlib.sha256()
    for key in sorted(index):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update((index[key].get("hash") or "").encode())
        digest.update(b"\0")
    for key in sorted(pending or {}):
        digest.update(b"pending\0" + key.encode() + b"\0" + str(pending[key].get("state")).encode() + b"\0")
    if not ready:
        digest.update(b"starting")
    return digest.hexdigest()[:16]


def _refresh_version() -> None:
    """Caller holds _LOCK."""
    global _VERSION
    _VERSION = _compute_version(_INDEX, _PENDING, _READY)


def _remember(entry: Dict) -> None:
    """Add/replace an entry in the in-memory index and refresh the version."""
    with _LOCK:
        _INDEX[entry["key"]] = entry
        _PENDING.pop(entry["key"], None)
        _refresh_version()


def _key_lock(key: str) -> threading.Lock:
    with _LOCK:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


def preload() -> int:
    """
    Publish every sidecar on disk at once, before any model file is listed or read. Entries under
    old keys (a moved file, the old extra-path layout) are served too until the first scan has
    listed the files: the app matches a model by the end of its key, so they still give it its
    metadata. Returns how many entries were loaded.
    """
    sidecars = {k: e for k, e in _load_sidecars().items() if _valid(e)}
    with _LOCK:
        for key, entry in sidecars.items():
            _INDEX.setdefault(key, entry)
        _refresh_version()
    return len(sidecars)


def hash_model(filepath: str, key: Optional[str] = None) -> Dict:
    """
    Ensure a model file is hashed. Returns its index dict. If the model has
    already been hashed and its size/mtime match, this is a cheap no-op.
    Either way the result is recorded in the in-memory index.
    """
    if key is None:
        key = cache_key(filepath)

    # Downloads and the scanner can discover the same file simultaneously.
    with _key_lock(key):
        if os.path.exists(filepath + ".aria2"):
            raise ValueError("Download still in progress")
        signature = _file_signature(filepath)
        index = load_index(key)
        if not (index and index.get("hash") and index.get("file") == signature):
            print(f"[ImageLab] Hashing {key} ...")
            digest = calculate_sha256(filepath)
            if os.path.exists(filepath + ".aria2") or _file_signature(filepath) != signature:
                raise ValueError("File changed while hashing; retrying on next scan")
            index = {
                "key": key,
                "filename": os.path.basename(filepath),
                "hash": digest,
                "hashed_at": time.time(),
                "file": signature,
            }
            save_index(key, index)

        _remember(index)
        return index


def _tail_matches(old_key: str, new_key: str) -> bool:
    """`C/Users/x/models/checkpoints/a.safetensors` ends in `checkpoints/a.safetensors`."""
    return old_key == new_key or old_key.endswith("/" + new_key)


def _find_moved(key: str, signature: Dict, by_name: Dict[str, List[Dict]]) -> Optional[Dict]:
    """
    A sidecar filed under another key for this very file: same file name, size and mtime_ns
    (a rename or move keeps all three). Prefers one whose key ends in the new key (the old
    extra-path layout), then any other. Returns None when nothing matches.
    """
    filename = key.rsplit("/", 1)[-1]
    candidates = [
        e for e in by_name.get(filename.lower(), ())
        if e["key"] != key and e.get("file") == signature
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda e: (not _tail_matches(e["key"], key), e["key"]))
    return candidates[0]


def _list_models() -> Dict[str, Tuple[str, str]]:
    """Every model file ComfyUI can see: cache_key -> (folder type, path). One entry per file."""
    files: Dict[str, Tuple[str, str]] = {}
    for model_type in MODEL_TYPES:
        if model_type not in folder_paths.folder_names_and_paths:
            continue
        for filename in folder_paths.get_filename_list(model_type):
            filepath = folder_paths.get_full_path(model_type, filename)
            if not filepath:
                continue
            files.setdefault(cache_key(filepath), (model_type, filepath))
    return files


def build_index() -> Dict[str, int]:
    """
    Scan every model folder, hash anything not yet hashed, and populate the
    in-memory index from the on-disk cache. Safe to repeat after downloads.
    Returns a per-type count of files newly hashed this run.

    1. List the files and reconcile them with the sidecars: a matching sidecar under the file's
       key is used; one under an old key (same name + size + mtime_ns) is copied to the new key
       (the old file is kept). Everything known is published at once; keys of files that are gone
       drop out of memory (their sidecars stay, for when a share comes back).
    2. Hash what is left, smallest first, each published the moment it is done.
    """
    global _READY
    counts: Dict[str, int] = {}

    with _LOCK:
        known_before = set(_INDEX)
    sidecars = _load_sidecars()
    by_name: Dict[str, List[Dict]] = {}
    for entry in sidecars.values():
        if _valid(entry):
            by_name.setdefault(entry["key"].rsplit("/", 1)[-1].lower(), []).append(entry)

    found: Dict[str, Dict] = {}
    todo: List[Tuple[int, str, str, str]] = []  # (size, key, model type, path)
    pending: Dict[str, Dict] = {}
    for key, (model_type, filepath) in _list_models().items():
        try:
            # aria2 writes to the final filename before it finishes.
            if os.path.exists(filepath + ".aria2"):
                forget(filepath)
                continue
            signature = _file_signature(filepath)
            entry = sidecars.get(key)
            if _valid(entry) and entry["file"] == signature:
                found[key] = entry
                continue
            moved = _find_moved(key, signature, by_name)
            if moved:
                entry = dict(moved, key=key, filename=os.path.basename(filepath))
                save_index(key, entry)
                print(f"[ImageLab] Hash of {key} carried over from {moved['key']}")
                found[key] = entry
                continue
            if entry is not None:
                forget(filepath)  # replaced file, or a sidecar without a signature: hash again
            todo.append((signature["size"], key, model_type, filepath))
            pending[key] = {"key": key, "filename": os.path.basename(filepath), "folder": model_type,
                            "size": signature["size"], "state": "queued"}
        except Exception as e:
            print(f"[ImageLab] Error checking {filepath}: {e}")

    with _LOCK:
        listed = set(found) | set(pending)
        for key in known_before - listed:
            # Gone from disk, or an old key that a listed file now carries. A key a download
            # published while we were listing is not in known_before, so it stays.
            _INDEX.pop(key, None)
        _INDEX.update(found)
        # Keep a key a download is hashing right now in the list; drop pending ones that vanished.
        _PENDING.clear()
        _PENDING.update({k: v for k, v in pending.items() if k not in _INDEX})
        _READY = True
        _refresh_version()

    for size, key, model_type, filepath in sorted(todo):
        with _LOCK:
            if key not in _PENDING:
                continue  # a download hashed it meanwhile
            _PENDING[key] = dict(_PENDING[key], state="hashing")
            _refresh_version()
        _CURRENT.clear()
        _CURRENT.update({"path": filepath, "key": key, "filename": os.path.basename(filepath),
                         "size": size, "done": 0})
        try:
            hash_model(filepath, key)
            counts[model_type] = counts.get(model_type, 0) + 1
        except Exception as e:
            print(f"[ImageLab] Error hashing {key}: {e}")
            with _LOCK:
                if key in _PENDING:
                    _PENDING[key] = dict(_PENDING[key], state="error", error=str(e))
                    _refresh_version()
        finally:
            _CURRENT.clear()

    return counts


def watch_models() -> None:
    """Downloads can finish long after ComfyUI starts; keep discovering them."""
    while True:
        try:
            counts = build_index()
            if counts:
                print(f"[ImageLab] Hashed: {counts}")
        except Exception as e:
            print(f"[ImageLab] Error scanning models: {e}")
        time.sleep(30)


def get_snapshot() -> Tuple[str, List[Dict]]:
    """
    Return `(version, entries)` for the API layer. `version` is an ETag-style
    content token; `entries` is the list of `{key, filename, hash, hashed_at}`
    dicts. Cheap — a lock-guarded copy, no disk I/O.
    """
    with _LOCK:
        return _VERSION, list(_INDEX.values())


def _progress() -> Dict:
    """Caller holds _LOCK."""
    return {
        "hashed": len(_INDEX),
        "total": len(_INDEX) + len(_PENDING),
        "pending": len(_PENDING),
        "ready": _READY,
    }


def get_listing() -> Tuple[str, List[Dict], List[Dict], Dict]:
    """
    `(version, entries, pending, progress)` in one consistent read. `pending` lists the models
    on disk without a hash yet (`{key, filename, folder, size, state}`, smallest first — the
    order they are hashed in); `progress` is `{hashed, total, pending, ready}`. `ready` is false
    until the first scan has listed every file: until then `entries` are the preloaded sidecars.
    """
    with _LOCK:
        pending = sorted(_PENDING.values(), key=lambda p: (p.get("size") or 0, p["key"]))
        return _VERSION, list(_INDEX.values()), pending, _progress()


def get_status() -> Dict:
    """Progress plus the file being read right now (with bytes done). Not part of the ETag."""
    with _LOCK:
        status = _progress()
    current = dict(_CURRENT)
    current.pop("path", None)
    status["current"] = current or None
    return status


def forget(filepath: str) -> None:
    """
    Drop a model from the index: delete its `.index` sidecar and remove its
    in-memory entry. Called when a model file is deleted from disk. `cache_key`
    is pure path math, so this works whether or not the file still exists.
    """
    key = cache_key(filepath)
    path = os.path.join(INDEX_DIR, *key.split("/")) + ".index"
    if os.path.exists(path):
        os.remove(path)
    with _LOCK:
        removed = _INDEX.pop(key, None) is not None
        removed = _PENDING.pop(key, None) is not None or removed
        if removed:
            _refresh_version()


def relocate(old_path: str, new_path: str) -> None:
    """
    A model file was renamed on the same disk (`model_sort` moving it to diffusion_models/):
    carry its hash over to the new path instead of reading gigabytes again. Size and mtime
    survive a rename, so the signature still matches. Falls back to a fresh hash if the old
    entry is missing or stale.
    """
    old_key, new_key = cache_key(old_path), cache_key(new_path)
    previous = load_index(old_key)
    forget(old_path)
    if previous and previous.get("hash") and previous.get("file") == _file_signature(new_path):
        entry = dict(previous, key=new_key, filename=os.path.basename(new_path))
        save_index(new_key, entry)
        _remember(entry)
    else:
        hash_model(new_path, new_key)
