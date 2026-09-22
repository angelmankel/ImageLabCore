"""
Favorites — persist explicitly starred images on disk so they survive
ComfyUI restarts (PreviewImage outputs in `temp/` are wiped on next boot).

Storage layout:

    <comfy_output>/imagelab_favorites/<MM-DD-YYYY>/<filename>

The MM-DD-YYYY folder is materialised lazily on first save of the day. Files
keep their source filename when possible; on collision a numeric suffix is
inserted before the extension so two clients can favorite different images
with the same default name without clobbering each other.

Endpoints (registered from api.py):

    POST   /imagelab/favorites              body: {filename, subfolder?, type?}
    GET    /imagelab/favorites              -> {version, favorites:[...]}, ETag'd
    DELETE /imagelab/favorites/{date}/{filename}
    GET    /imagelab/favorites/view?date=&filename=

The list endpoint is cheap to poll — it just stats the favorites dir tree
and hashes the resulting (path, mtime) tuples into a content version that
clients can `If-None-Match` against.
"""
import hashlib
import mimetypes
import os
import re
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import folder_paths


# Subdirectory under ComfyUI's output dir where favorites live. Kept inside
# `output/` (a) so users browsing the file system find them next to their
# generations, and (b) so the dir is on the same volume in RunPod / Docker
# setups (no cross-device-link surprises).
FAVORITES_SUBDIR = "imagelab_favorites"

# Filename safety — strip path separators and weird control chars before
# joining anything to disk. Anything outside this set is replaced with `_`.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

# MM-DD-YYYY folder names — matches the user-facing convention.
_DATE_FMT = "%m-%d-%Y"
_DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")


# ─── path helpers ──────────────────────────────────────────────────────────


def _favorites_root() -> str:
    """Absolute path of the favorites root, creating it on first call."""
    root = os.path.join(folder_paths.get_output_directory(), FAVORITES_SUBDIR)
    os.makedirs(root, exist_ok=True)
    return root


def _today_folder() -> str:
    """The MM-DD-YYYY folder name for today (server local time)."""
    return datetime.now().strftime(_DATE_FMT)


def _safe_filename(name: str) -> str:
    """Sanitise a filename — keep only `[A-Za-z0-9._-]`, collapse anything else."""
    name = os.path.basename(name).strip() or "image.png"
    return _SAFE_NAME.sub("_", name) or "image.png"


def _resolve_source(filename: str, subfolder: str, image_type: str) -> Optional[str]:
    """
    Resolve a comfy `/view`-style image reference to an on-disk path.

    `image_type` is one of `output` / `temp` / `input` (ComfyUI's three
    canonical roots). We refuse anything else so a crafted request can't
    read arbitrary files on the host.
    """
    if image_type == "output":
        base = folder_paths.get_output_directory()
    elif image_type == "temp":
        base = folder_paths.get_temp_directory()
    elif image_type == "input":
        base = folder_paths.get_input_directory()
    else:
        return None

    candidate = os.path.normpath(os.path.join(base, subfolder or "", filename))
    base_real = os.path.realpath(base)
    cand_real = os.path.realpath(candidate)
    # Path traversal guard: the resolved path must stay inside the chosen base.
    if not (cand_real == base_real or cand_real.startswith(base_real + os.sep)):
        return None
    if not os.path.isfile(cand_real):
        return None
    return cand_real


def _unique_path(folder: str, name: str) -> str:
    """If `folder/name` exists, insert `-1`, `-2`, … before the extension."""
    target = os.path.join(folder, name)
    if not os.path.exists(target):
        return target
    stem, ext = os.path.splitext(name)
    i = 1
    while True:
        candidate = os.path.join(folder, f"{stem}-{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


# ─── snapshot + ETag ───────────────────────────────────────────────────


def _list_entries() -> List[Dict]:
    """Walk the favorites tree once, return one dict per favorite."""
    root = _favorites_root()
    out: List[Dict] = []
    try:
        for date_folder in sorted(os.listdir(root)):
            if not _DATE_RE.match(date_folder):
                continue
            date_path = os.path.join(root, date_folder)
            if not os.path.isdir(date_path):
                continue
            for entry in sorted(os.listdir(date_path)):
                path = os.path.join(date_path, entry)
                if not os.path.isfile(path):
                    continue
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                out.append({
                    "id": f"{date_folder}/{entry}",
                    "date": date_folder,
                    "filename": entry,
                    "size": st.st_size,
                    "saved_at": st.st_mtime,
                })
    except FileNotFoundError:
        pass
    return out


def _snapshot() -> Tuple[str, List[Dict]]:
    """Return (etag, entries). ETag changes whenever any file's (path, mtime, size) changes."""
    entries = _list_entries()
    h = hashlib.sha256()
    for e in entries:
        h.update(f"{e['id']}|{e['saved_at']}|{e['size']}".encode("utf-8"))
    return h.hexdigest()[:16], entries


# ─── public API (called by api.py) ─────────────────────────────────────────


def save_favorite(filename: str, subfolder: str = "", image_type: str = "temp") -> Dict:
    """
    Copy a comfy-managed image into the favorites folder for today.

    Raises FileNotFoundError if the source can't be resolved or doesn't exist.
    Returns the new favorite's dict (same shape as `list_favorites()` entries).
    """
    source = _resolve_source(filename, subfolder, image_type)
    if not source:
        raise FileNotFoundError(f"Source image not found: type={image_type} subfolder={subfolder!r} filename={filename!r}")

    date = _today_folder()
    date_dir = os.path.join(_favorites_root(), date)
    os.makedirs(date_dir, exist_ok=True)

    safe = _safe_filename(filename)
    target = _unique_path(date_dir, safe)
    # copy2 preserves mtime so saved_at matches the source's generation time
    # when possible — meaningful when a user favorites an older generation.
    shutil.copy2(source, target)

    name = os.path.basename(target)
    try:
        size = os.path.getsize(target)
        mtime = os.path.getmtime(target)
    except OSError:
        size = 0
        mtime = time.time()
    return {
        "id": f"{date}/{name}",
        "date": date,
        "filename": name,
        "size": size,
        "saved_at": mtime,
    }


def list_favorites() -> Tuple[str, List[Dict]]:
    """Public snapshot accessor — returns (etag, entries)."""
    return _snapshot()


def delete_favorite(date: str, filename: str) -> bool:
    """Remove a favorite. Returns True if it existed, False otherwise."""
    if not _DATE_RE.match(date or ""):
        return False
    safe = _safe_filename(filename)
    # Build the path through the same realpath guard the resolver uses so a
    # crafted "../" can't escape the favorites root.
    root = _favorites_root()
    candidate = os.path.normpath(os.path.join(root, date, safe))
    root_real = os.path.realpath(root)
    cand_real = os.path.realpath(candidate)
    if not cand_real.startswith(root_real + os.sep):
        return False
    if not os.path.isfile(cand_real):
        return False
    try:
        os.remove(cand_real)
    except OSError:
        return False
    # Best-effort: prune the date folder if it just emptied. Idempotent.
    try:
        os.rmdir(os.path.dirname(cand_real))
    except OSError:
        pass
    return True


def open_favorite(date: str, filename: str) -> Optional[Tuple[str, str]]:
    """Resolve (date, filename) to (absolute path, content-type) for serving."""
    if not _DATE_RE.match(date or ""):
        return None
    safe = _safe_filename(filename)
    root = _favorites_root()
    candidate = os.path.normpath(os.path.join(root, date, safe))
    root_real = os.path.realpath(root)
    cand_real = os.path.realpath(candidate)
    if not cand_real.startswith(root_real + os.sep):
        return None
    if not os.path.isfile(cand_real):
        return None
    mime, _ = mimetypes.guess_type(cand_real)
    return cand_real, (mime or "application/octet-stream")
